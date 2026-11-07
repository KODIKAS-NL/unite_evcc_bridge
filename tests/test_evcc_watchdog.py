"""Tests for the evcc watchdog.

The charger's failsafe only fires when Home Assistant stops writing the alive
heartbeat. When evcc itself goes silent, HA keeps the heartbeat going, so the
bridge has to cap the current itself. The rules under test:

* evcc silent for the timeout -> charge current capped at the failsafe current
  on the wire, evcc's intent kept.
* While capped, every write path (evcc commands, session start, reconnect,
  phase-recovery resume) respects the cap.
* evcc back -> its last intent is written again (evcc won't re-send it).
* Sources: API (/api/health, falling back to /api/state) or MQTT, where
  retained messages are not a sign of life.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web

import custom_components.unite_evcc_bridge.coordinator as coordinator_module
from custom_components.unite_evcc_bridge.coordinator import WebastoEvccCoordinator
from custom_components.unite_evcc_bridge.evcc_watchdog import (
    TRANSITION_LOST,
    TRANSITION_RECOVERED,
    EvccApiError,
    EvccLiveness,
    EvccWatchdog,
    async_probe_evcc_api,
    normalize_evcc_url,
    probe_interval_s,
)
from custom_components.unite_evcc_bridge.models import ChargerSnapshot


# --- pure liveness logic -----------------------------------------------------


def test_liveness_grace_then_lost_then_recovered() -> None:
    live = EvccLiveness(timeout_s=120, last_seen=1000.0)
    assert live.evaluate(1100.0) is None  # still within the start-up grace
    assert live.evaluate(1120.0) == TRANSITION_LOST
    assert live.evaluate(1500.0) is None  # edge, not level: reported once
    live.mark_alive(1501.0)
    assert live.evaluate(1502.0) == TRANSITION_RECOVERED
    assert live.evaluate(1503.0) is None


def test_liveness_never_moves_backwards() -> None:
    live = EvccLiveness(timeout_s=60, last_seen=500.0)
    live.mark_alive(400.0)
    assert live.last_seen == 500.0


def test_normalize_url_and_probe_interval() -> None:
    assert normalize_evcc_url("evcc:7070/") == "http://evcc:7070"
    assert normalize_evcc_url(" https://evcc.lan ") == "https://evcc.lan"
    assert normalize_evcc_url("") == ""
    assert probe_interval_s(30) == 7
    assert probe_interval_s(120) == 15
    assert probe_interval_s(10) == 5


# --- MQTT source ----------------------------------------------------------------


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_mqtt_live_message_marks_alive_retained_does_not() -> None:
    clock = Clock()
    wd = EvccWatchdog(None, mode="mqtt", timeout_s=60, mqtt_topic="evcc/updated", clock=clock)
    clock.now += 100
    assert wd.is_stale()

    # A retained message is the broker replaying history, not evcc talking.
    wd.handle_mqtt_message(SimpleNamespace(payload="1730000000", retain=True))
    assert wd.is_stale()

    wd.handle_mqtt_message(SimpleNamespace(payload="1730000010", retain=False))
    assert not wd.is_stale()


# --- API source -----------------------------------------------------------------


def _run_with_server(routes: dict[str, int], body) -> object:
    """Start a tiny aiohttp server with the given path -> status map."""

    async def _main():
        app = web.Application()
        for path, status in routes.items():
            async def handler(request, status=status):
                return web.Response(status=status, text="OK")
            app.router.add_get(path, handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                return await body(session, f"127.0.0.1:{port}")
        finally:
            await runner.cleanup()

    return asyncio.run(_main())


def test_api_probe_prefers_health() -> None:
    async def body(session, url):
        return await async_probe_evcc_api(session, url)

    assert _run_with_server({"/api/health": 200, "/api/state": 200}, body) == "/api/health"


def test_api_probe_falls_back_to_state_on_404() -> None:
    async def body(session, url):
        return await async_probe_evcc_api(session, url)

    assert _run_with_server({"/api/state": 200}, body) == "/api/state"


def test_api_probe_unhealthy_evcc_is_not_alive() -> None:
    async def body(session, url):
        with pytest.raises(EvccApiError):
            await async_probe_evcc_api(session, url)
        return True

    assert _run_with_server({"/api/health": 503, "/api/state": 200}, body)


def test_api_probe_unreachable_is_not_alive() -> None:
    async def _main():
        async with aiohttp.ClientSession() as session:
            with pytest.raises(EvccApiError):
                await async_probe_evcc_api(session, "http://127.0.0.1:1", timeout_s=2)

    asyncio.run(_main())


def test_watchdog_api_probe_marks_alive_and_remembers_path() -> None:
    clock = Clock()

    async def body(session, url):
        wd = EvccWatchdog(None, mode="api", timeout_s=60, api_url=url, session=session, clock=clock)
        clock.now += 100
        assert wd.is_stale()
        assert await wd.async_probe_api()
        assert not wd.is_stale()
        assert wd._api_path == "/api/state"
        assert wd.last_error is None
        return True

    assert _run_with_server({"/api/state": 200}, body)


# --- coordinator behaviour ----------------------------------------------------


class FakeClient:
    def __init__(self) -> None:
        self.writes: list[tuple[str, int]] = []
        self.stats = SimpleNamespace(connected=True, connection_epoch=0)
        self.snapshot = ChargerSnapshot(available=True, cable_state=2, charging_state=1)

    async def write(self, register, value) -> None:
        self.writes.append((register.name, value))

    async def read_snapshot(self) -> ChargerSnapshot:
        return self.snapshot


def _setup(monkeypatch, *, failsafe_current: int = 6):
    client = FakeClient()
    hass = SimpleNamespace(async_create_task=lambda coro: coro.close())
    entry = SimpleNamespace(entry_id="test", options={}, data={"host": "192.0.2.1"})
    coord = WebastoEvccCoordinator(
        hass,
        entry=entry,
        client=client,
        poll_interval=10,
        max_current=16,
        failsafe_current=failsafe_current,
        failsafe_timeout=30,
        phase_recovery_enabled=False,
        phase_recovery_observe=60,
        phase_recovery_dwell=121,
    )
    clock = Clock()
    coord.evcc_watchdog = EvccWatchdog(
        hass, mode="mqtt", timeout_s=120, mqtt_topic="evcc/updated", clock=clock
    )
    # Steady state: connection owned, vehicle already plugged in.
    coord._initialized_connection_epoch = 0
    coord._vehicle_was_connected = True
    coord.requested_phase = "3"

    async def _noop_heartbeat(client) -> None:
        return None

    async def _noop_guard(data) -> None:
        return None

    monkeypatch.setattr(coordinator_module, "write_heartbeat", _noop_heartbeat)
    monkeypatch.setattr(coord, "_guard_phase_setting", _noop_guard)
    return coord, client, clock


def _poll(coord) -> None:
    asyncio.run(coord._async_update_data())


def test_evcc_lost_caps_current_and_keeps_intent(monkeypatch) -> None:
    coord, client, clock = _setup(monkeypatch)
    asyncio.run(coord.async_set_current(16))
    client.writes.clear()

    clock.now += 60
    _poll(coord)
    assert client.writes == []  # within the timeout: nothing happens

    clock.now += 61
    _poll(coord)
    assert coord.evcc_throttled
    assert client.writes == [("current_limit", 6)]
    assert coord.current_intent == 16  # evcc's intent is untouched

    client.writes.clear()
    _poll(coord)
    assert client.writes == []  # capped once, not on every poll


def test_evcc_back_restores_intent(monkeypatch) -> None:
    coord, client, clock = _setup(monkeypatch)
    asyncio.run(coord.async_set_current(16))
    clock.now += 200
    _poll(coord)
    client.writes.clear()

    coord.evcc_watchdog.handle_mqtt_message(SimpleNamespace(retain=False))
    _poll(coord)
    assert not coord.evcc_throttled
    assert client.writes == [("current_limit", 16)]


def test_cap_never_raises_a_lower_intent(monkeypatch) -> None:
    coord, client, clock = _setup(monkeypatch)
    asyncio.run(coord.async_set_current(0))
    clock.now += 200
    client.writes.clear()
    _poll(coord)
    assert client.writes == [("current_limit", 0)]


def test_zero_failsafe_stops_charging_when_evcc_lost(monkeypatch) -> None:
    coord, client, clock = _setup(monkeypatch, failsafe_current=0)
    asyncio.run(coord.async_set_current(16))
    clock.now += 200
    client.writes.clear()
    _poll(coord)
    assert client.writes == [("current_limit", 0)]


def test_reconnect_while_lost_programs_capped_current(monkeypatch) -> None:
    coord, client, clock = _setup(monkeypatch)
    asyncio.run(coord.async_set_current(16))
    clock.now += 200
    client.stats.connection_epoch = 1  # a new Modbus connection this poll
    coord._baseline = {}
    coord._baseline_loaded = True

    async def _noop_resync() -> None:
        return None

    monkeypatch.setattr(coord, "_async_resync_phase_after_reconnect", _noop_resync)
    client.writes.clear()
    _poll(coord)
    current_writes = [v for name, v in client.writes if name == "current_limit"]
    assert current_writes and max(current_writes) == 6


def test_session_start_while_lost_is_capped(monkeypatch) -> None:
    coord, client, clock = _setup(monkeypatch)
    asyncio.run(coord.async_set_current(16))
    clock.now += 200
    _poll(coord)  # lost
    coord._vehicle_was_connected = False  # next poll sees a fresh plug-in

    async def _noop_phase(*, context) -> None:
        return None

    monkeypatch.setattr(coord, "_async_reassert_phase", _noop_phase)
    client.writes.clear()
    _poll(coord)
    assert client.writes == [("current_limit", 6)]


def test_evcc_command_counts_as_sign_of_life(monkeypatch) -> None:
    coord, client, clock = _setup(monkeypatch)
    asyncio.run(coord.async_set_current(16))
    clock.now += 200
    _poll(coord)
    assert coord.evcc_throttled

    # evcc reached HA, so it is alive. The new value is written as asked,
    # once the next poll notices the recovery.
    asyncio.run(coord.async_set_current(10))
    client.writes.clear()
    _poll(coord)
    assert not coord.evcc_throttled
    assert client.writes == [("current_limit", 10)]


def test_no_watchdog_changes_nothing(monkeypatch) -> None:
    coord, client, clock = _setup(monkeypatch)
    coord.evcc_watchdog = None
    asyncio.run(coord.async_set_current(16))
    client.writes.clear()
    clock.now += 10_000
    _poll(coord)
    assert client.writes == []
    assert not coord.evcc_throttled
