"""evcc liveness watchdog.

The charger's own failsafe only covers the Home Assistant <-> charger link:
Home Assistant writes the alive heartbeat on every poll, whether or not evcc
is still around. If evcc crashes or loses its network, the charger would keep
charging at evcc's last setpoint indefinitely.

This module watches an independent sign of life from evcc - its HTTP API or
an MQTT topic it publishes every cycle - and reports when that goes stale, so
the coordinator can cap the charge current at the failsafe current until evcc
is back. evcc does not re-send an unchanged setpoint, so on recovery the
coordinator re-writes evcc's last intent itself.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

import aiohttp

_LOGGER = logging.getLogger(__name__)

WATCHDOG_OFF = "off"
WATCHDOG_API = "api"
WATCHDOG_MQTT = "mqtt"
WATCHDOG_MODES = (WATCHDOG_OFF, WATCHDOG_API, WATCHDOG_MQTT)

TRANSITION_LOST = "lost"
TRANSITION_RECOVERED = "recovered"

# Tried in order. /api/health reflects whether evcc's control loop is healthy;
# /api/state is the fallback for builds without the health endpoint.
API_PATHS = ("/api/health", "/api/state")
API_REQUEST_TIMEOUT_S = 5


class EvccApiError(Exception):
    """evcc's API could not be reached or answered with an error."""


def normalize_evcc_url(url: str) -> str:
    """'evcc:7070' -> 'http://evcc:7070'; trailing slashes dropped."""
    url = (url or "").strip().rstrip("/")
    if url and "://" not in url:
        url = f"http://{url}"
    return url


def probe_interval_s(timeout_s: int) -> int:
    """Check a few times per timeout window, but not more often than every 5 s."""
    return max(5, min(15, int(timeout_s) // 4))


async def async_probe_evcc_api(
    session: aiohttp.ClientSession,
    base_url: str,
    *,
    paths: tuple[str, ...] = API_PATHS,
    timeout_s: float = API_REQUEST_TIMEOUT_S,
) -> str:
    """Return the API path that answered 200. Raise EvccApiError otherwise.

    A 404 moves on to the next path; any other non-200 status (an unhealthy
    evcc answers /api/health with 5xx) counts as "not alive".
    """
    base = normalize_evcc_url(base_url)
    if not base:
        raise EvccApiError("no evcc URL configured")
    for path in paths:
        try:
            async with session.get(
                f"{base}{path}", timeout=aiohttp.ClientTimeout(total=timeout_s)
            ) as resp:
                if resp.status == 404:
                    continue
                await resp.read()
                if resp.status == 200:
                    return path
                raise EvccApiError(f"{path} answered HTTP {resp.status}")
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as err:
            raise EvccApiError(f"{base}{path}: {err or type(err).__name__}") from err
    raise EvccApiError(f"no evcc API found at {base}")


@dataclass
class EvccLiveness:
    """Pure stale/fresh bookkeeping on a monotonic clock.

    ``last_seen`` starts at construction time, so a missing evcc is only
    declared lost after one full timeout (a grace period while both boot).
    """

    timeout_s: float
    last_seen: float
    lost: bool = False

    def mark_alive(self, now: float) -> None:
        self.last_seen = max(self.last_seen, now)

    def is_stale(self, now: float) -> bool:
        return now - self.last_seen >= self.timeout_s

    def seconds_since_seen(self, now: float) -> float:
        return max(0.0, now - self.last_seen)

    def evaluate(self, now: float) -> str | None:
        """Return TRANSITION_LOST / TRANSITION_RECOVERED on an edge, else None."""
        stale = self.is_stale(now)
        if stale and not self.lost:
            self.lost = True
            return TRANSITION_LOST
        if not stale and self.lost:
            self.lost = False
            return TRANSITION_RECOVERED
        return None


class EvccWatchdog:
    """Feeds EvccLiveness from evcc's HTTP API or from an MQTT topic."""

    def __init__(
        self,
        hass: Any,
        *,
        mode: str,
        timeout_s: int,
        api_url: str | None = None,
        mqtt_topic: str | None = None,
        session: aiohttp.ClientSession | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if mode not in (WATCHDOG_API, WATCHDOG_MQTT):
            raise ValueError(f"unsupported evcc watchdog mode: {mode}")
        self.hass = hass
        self.mode = mode
        self.timeout_s = int(timeout_s)
        self.api_url = normalize_evcc_url(api_url or "")
        self.mqtt_topic = (mqtt_topic or "").strip()
        self._session = session
        self._clock = clock
        self.liveness = EvccLiveness(timeout_s=self.timeout_s, last_seen=clock())
        self.last_error: str | None = None
        self._api_path: str | None = None  # remembered after the first success
        self._unsub_timer: Callable[[], None] | None = None
        self._unsub_mqtt: Callable[[], None] | None = None
        self._stopped = False

    # -- status --------------------------------------------------------------

    @property
    def source(self) -> str:
        return self.api_url if self.mode == WATCHDOG_API else self.mqtt_topic

    def is_stale(self) -> bool:
        return self.liveness.is_stale(self._clock())

    def seconds_since_seen(self) -> float:
        return self.liveness.seconds_since_seen(self._clock())

    def evaluate(self) -> str | None:
        return self.liveness.evaluate(self._clock())

    def mark_alive(self) -> None:
        self.liveness.mark_alive(self._clock())

    # -- lifecycle -----------------------------------------------------------

    async def async_start(self) -> None:
        from homeassistant.helpers.event import async_track_time_interval

        self._unsub_timer = async_track_time_interval(
            self.hass,
            self._async_tick,
            timedelta(seconds=probe_interval_s(self.timeout_s)),
        )
        await self._async_tick()

    async def async_stop(self) -> None:
        self._stopped = True
        if self._unsub_timer is not None:
            self._unsub_timer()
            self._unsub_timer = None
        if self._unsub_mqtt is not None:
            self._unsub_mqtt()
            self._unsub_mqtt = None

    async def _async_tick(self, _now: Any = None) -> None:
        if self._stopped:
            return
        try:
            if self.mode == WATCHDOG_API:
                await self.async_probe_api()
            elif self._unsub_mqtt is None:
                # (Re)try until Home Assistant's MQTT client is up.
                await self._async_subscribe_mqtt()
        except Exception as err:  # noqa: BLE001 - a probe must never kill the timer
            self.last_error = str(err)
            _LOGGER.debug("evcc watchdog tick failed: %s", err)

    # -- API source ----------------------------------------------------------

    async def async_probe_api(self) -> bool:
        if self._session is None:
            from homeassistant.helpers.aiohttp_client import async_get_clientsession

            self._session = async_get_clientsession(self.hass)
        paths = (self._api_path,) if self._api_path else API_PATHS
        try:
            self._api_path = await async_probe_evcc_api(
                self._session, self.api_url, paths=paths
            )
        except EvccApiError as err:
            self.last_error = str(err)
            _LOGGER.debug("evcc API probe failed: %s", err)
            return False
        self.last_error = None
        self.mark_alive()
        return True

    # -- MQTT source ---------------------------------------------------------

    async def _async_subscribe_mqtt(self) -> None:
        from homeassistant.components import mqtt
        from homeassistant.core import callback

        if not await mqtt.async_wait_for_mqtt_client(self.hass):
            self.last_error = "Home Assistant's MQTT integration is not available"
            _LOGGER.debug(self.last_error)
            return

        @callback
        def _on_message(msg: Any) -> None:  # runs in the event loop
            self.handle_mqtt_message(msg)

        self._unsub_mqtt = await mqtt.async_subscribe(
            self.hass, self.mqtt_topic, _on_message, qos=0
        )
        self.last_error = None
        _LOGGER.debug("evcc watchdog subscribed to MQTT topic %s", self.mqtt_topic)

    def handle_mqtt_message(self, msg: Any) -> None:
        """Any live message on the topic counts as a sign of life.

        Retained messages are ignored: evcc publishes with retain, so on
        (re)subscribe the broker hands over the last value even when evcc
        has been dead for hours. Per the MQTT spec a message forwarded live
        to an existing subscriber never carries the retain flag.
        """
        if getattr(msg, "retain", False):
            return
        self.mark_alive()
