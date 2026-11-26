"""Bounded asynchronous HTTP requests with quota-scoped pacing and retries."""

import asyncio
import math
import random
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
_REPLAYABLE_METHODS = frozenset({"GET", "HEAD"})
_TRANSIENT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.RemoteProtocolError,
)
_BACKOFF_BASE = 1.0
_BACKOFF_CAP = 8.0


class Quota:
    """Shared pacing state for one documented provider quota scope.

    Callers pass one instance to every request in the same scope, including
    retries. Instances belong to a single event loop and hold one monotonic
    next-start time; they make no assumptions about hosts, methods or endpoints.
    """

    def __init__(self, min_interval: float = 0.0) -> None:
        if not math.isfinite(min_interval) or min_interval < 0.0:
            raise ValueError("min_interval must be finite and non-negative")
        self.min_interval = min_interval
        self._next_start = 0.0

    async def _wait(self) -> None:
        """Claim the next start slot, sleeping until it is due.

        The slot is claimed without an intervening await, so concurrent callers
        cannot take it, and a cooldown imposed while this caller slept is
        observed on the next check rather than bypassed.
        """
        while True:
            now = time.monotonic()
            if self._next_start <= now:
                self._next_start = now + self.min_interval
                return
            await asyncio.sleep(self._next_start - now)

    def _defer(self, delay: float) -> None:
        """Keep the scope from starting another attempt for delay seconds."""
        self._next_start = max(self._next_start, time.monotonic() + delay)


class HTTPClient:
    """Retrying requests over a caller-owned HTTPX client.

    The caller owns the AsyncClient lifetime, connection pool and per-operation
    timeouts, and supplies a client without transport retries or multi-request
    auth so that this class alone decides replays. Responses are buffered, so
    each attempt, including its body read, completes within the total timeout.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        total_timeout: float = 60.0,
        max_attempts: int = 2,
    ) -> None:
        if not math.isfinite(total_timeout) or total_timeout <= 0.0:
            raise ValueError("total_timeout must be finite and positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._client = client
        self._total_timeout = total_timeout
        self._max_attempts = max_attempts

    async def request(
        self,
        method: str,
        url: str,
        *,
        quota: Quota | None = None,
        replay_safe: bool | None = None,
        params: Mapping[str, str | int | float | bool | None] | None = None,
        headers: Mapping[str, str] | None = None,
        json: object = None,
    ) -> httpx.Response:
        """Send one request and return its final response.

        The total timeout bounds quota waits, every attempt with its buffered
        response, backoff and cooldowns, and raises TimeoutError when exceeded.
        Final transport failures re-raise and final HTTP responses are returned
        unexamined. Without a shared quota, cooldowns affect only this request.
        GET and HEAD replay by default. Set replay_safe explicitly to enable
        or disable replay for any method.
        """
        if replay_safe is None:
            replay_safe = method.upper() in _REPLAYABLE_METHODS
        max_attempts = self._max_attempts if replay_safe else 1
        if quota is None:
            quota = Quota()
        request = self._client.build_request(
            method, url, params=params, headers=headers, json=json
        )
        async with asyncio.timeout(self._total_timeout):
            for attempt in range(1, max_attempts):
                try:
                    response, deferred = await self._send_attempt(
                        request, quota, attempt
                    )
                except _TRANSIENT_ERRORS:
                    deferred = False
                else:
                    if response.status_code not in _RETRYABLE_STATUSES:
                        return response

                if not deferred:
                    await asyncio.sleep(_jitter_delay(attempt))

            response, _ = await self._send_attempt(request, quota, max_attempts)
            return response

    async def _send_attempt(
        self, request: httpx.Request, quota: Quota, attempt: int
    ) -> tuple[httpx.Response, bool]:
        """Apply any quota cooldown and report whether it governs the next retry."""
        await quota._wait()
        response = await self._client.send(
            request, stream=False, follow_redirects=False
        )
        if response.status_code not in (429, 503):
            return response, False

        cooldown = _retry_after(response)
        if cooldown is None and response.status_code == 429:
            cooldown = _jitter_delay(attempt)
        if cooldown is None:
            return response, False

        quota._defer(cooldown)
        return response, True


def _jitter_delay(attempt: int) -> float:
    """Full jitter below an exponential window capped at eight seconds."""
    window = min(_BACKOFF_CAP, _BACKOFF_BASE * 2.0 ** min(attempt - 1, 8))
    return random.uniform(0.0, window)


def _retry_after(response: httpx.Response) -> float | None:
    """Server-requested delay in seconds, or None when the header is unusable."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    value = value.strip()
    if value.isascii() and value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())
