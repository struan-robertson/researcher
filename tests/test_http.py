import asyncio
import itertools
import json
import math
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from researcher import http
from researcher.http import HTTPClient, Quota

_REAL_SLEEP = asyncio.sleep


class MockClock:
    """Advance monotonic time by each requested delay instead of waiting."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay
        await _REAL_SLEEP(0)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> MockClock:
    fake = MockClock()
    monkeypatch.setattr(http.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(http.asyncio, "sleep", fake.sleep)
    return fake


@pytest.fixture
def jitter(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Return zero delay and record the upper bound of each jitter window."""
    windows: list[float] = []

    def uniform(low: float, high: float) -> float:
        windows.append(high)
        return low

    monkeypatch.setattr(http.random, "uniform", uniform)
    return windows


def test_retry_replays_identical_request(jitter: list[float]) -> None:
    seen: list[tuple[str, str, bytes, dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            (request.method, str(request.url), request.content, dict(request.headers))
        )
        return httpx.Response(503 if len(seen) == 1 else 200)

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), headers={"x-session": "s1"}
        ) as client:
            return await HTTPClient(client).request(
                "POST",
                "https://api.test/search",
                params={"q": "graph search"},
                headers={"x-custom": "c"},
                json={"limit": 5},
                replay_safe=True,
            )

    response = asyncio.run(main())
    assert response.status_code == 200
    assert len(seen) == 2
    assert seen[0] == seen[1]
    method, url, body, headers = seen[0]
    assert method == "POST"
    assert url == "https://api.test/search?q=graph+search"
    assert json.loads(body) == {"limit": 5}
    assert headers["x-session"] == "s1"
    assert headers["x-custom"] == "c"


@pytest.mark.parametrize("method", ["GET", "HEAD", "get"])
def test_get_and_head_replay_by_default(method: str, jitter: list[float]) -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(503 if len(attempts) == 1 else 200)

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPClient(client).request(method, "https://api.test/")

    assert asyncio.run(main()).status_code == 200
    assert len(attempts) == 2


def test_post_does_not_replay_without_opt_in(jitter: list[float]) -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(503)

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPClient(client).request(
                "POST", "https://api.test/", json={"q": 1}
            )

    assert asyncio.run(main()).status_code == 503
    assert len(attempts) == 1


def test_replay_safe_false_disables_retries(jitter: list[float]) -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(503)

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPClient(client).request(
                "GET", "https://api.test/", replay_safe=False
            )

    assert asyncio.run(main()).status_code == 503
    assert len(attempts) == 1


@pytest.mark.parametrize("status", [400, 404, 409, 500, 501])
def test_permanent_status_is_returned_without_retry(
    status: int, jitter: list[float]
) -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(status)

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPClient(client).request("GET", "https://api.test/")

    response = asyncio.run(main())
    assert response.status_code == status
    assert len(attempts) == 1
    assert jitter == []


@pytest.mark.parametrize(
    "error_type", [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout]
)
def test_timeout_errors_are_retried(
    error_type: type[httpx.TimeoutException], jitter: list[float]
) -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            raise error_type("transient", request=request)
        return httpx.Response(200)

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPClient(client).request("GET", "https://api.test/")

    assert asyncio.run(main()).status_code == 200
    assert len(attempts) == 2


@pytest.mark.parametrize(
    "error_type",
    [httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError],
)
def test_transport_errors_are_retried(
    error_type: type[httpx.TransportError], jitter: list[float]
) -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            raise error_type("transient", request=request)
        return httpx.Response(200)

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPClient(client).request("GET", "https://api.test/")

    assert asyncio.run(main()).status_code == 200
    assert len(attempts) == 2


def test_final_transport_error_is_raised(jitter: list[float]) -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ConnectError("still down", request=request)

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await HTTPClient(client).request("GET", "https://api.test/")

    with pytest.raises(httpx.ConnectError):
        asyncio.run(main())
    assert len(attempts) == 2


@pytest.mark.parametrize(
    "error_type",
    [httpx.PoolTimeout, httpx.UnsupportedProtocol, httpx.LocalProtocolError],
)
def test_non_transient_errors_are_not_retried(
    error_type: type[httpx.RequestError], jitter: list[float]
) -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise error_type("permanent", request=request)

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await HTTPClient(client).request("GET", "https://api.test/")

    with pytest.raises(error_type):
        asyncio.run(main())
    assert len(attempts) == 1


def test_backoff_windows_are_capped(jitter: list[float]) -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(503)

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPClient(client, max_attempts=6).request(
                "GET", "https://api.test/"
            )

    assert asyncio.run(main()).status_code == 503
    assert len(attempts) == 6
    assert jitter == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_retry_after_seconds_delays_retry(clock: MockClock) -> None:
    attempts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(clock.now)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "3"})
        return httpx.Response(200)

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPClient(client).request("GET", "https://api.test/")

    assert asyncio.run(main()).status_code == 200
    assert clock.sleeps == [3.0]
    assert attempts[1] - attempts[0] == pytest.approx(3.0)


def test_retry_after_http_date_delays_retry(clock: MockClock) -> None:
    value = format_datetime(datetime.now(UTC) + timedelta(seconds=5), usegmt=True)
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(503, headers={"retry-after": value})
        return httpx.Response(200)

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPClient(client).request("GET", "https://api.test/")

    assert asyncio.run(main()).status_code == 200
    assert len(clock.sleeps) == 1
    assert 4.0 <= clock.sleeps[0] <= 5.0


def test_naive_retry_after_date_is_read_as_utc(clock: MockClock) -> None:
    when = datetime.now(UTC).replace(tzinfo=None) + timedelta(seconds=5)
    value = when.strftime("%a, %d %b %Y %H:%M:%S")
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": value})
        return httpx.Response(200)

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPClient(client).request("GET", "https://api.test/")

    assert asyncio.run(main()).status_code == 200
    assert 4.0 <= clock.sleeps[0] <= 5.0


def test_past_retry_after_date_retries_immediately(clock: MockClock) -> None:
    value = format_datetime(datetime.now(UTC) - timedelta(minutes=1), usegmt=True)
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": value})
        return httpx.Response(200)

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPClient(client).request("GET", "https://api.test/")

    assert asyncio.run(main()).status_code == 200
    assert not any(clock.sleeps)


def test_invalid_retry_after_uses_bounded_jitter(
    clock: MockClock, jitter: list[float]
) -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "soon"})
        return httpx.Response(200)

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPClient(client).request("GET", "https://api.test/")

    assert asyncio.run(main()).status_code == 200
    assert jitter == [1.0]
    assert not any(clock.sleeps)


def test_rate_limit_cooldown_is_shared_without_double_wait(
    clock: MockClock, jitter: list[float]
) -> None:
    quota = Quota()
    attempts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(clock.now)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "4"})
        return httpx.Response(200)

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPClient(client).request(
                "GET", "https://api.test/", quota=quota
            )

    assert asyncio.run(main()).status_code == 200
    assert attempts[1] - attempts[0] == pytest.approx(4.0)
    assert clock.sleeps == [4.0]


def test_service_unavailable_retry_after_defers_scope(clock: MockClock) -> None:
    quota = Quota()
    attempts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(clock.now)
        if len(attempts) == 1:
            return httpx.Response(503, headers={"retry-after": "5"})
        return httpx.Response(200)

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPClient(client).request(
                "GET", "https://api.test/", quota=quota
            )

    assert asyncio.run(main()).status_code == 200
    assert attempts[1] - attempts[0] == pytest.approx(5.0)
    assert clock.sleeps == [5.0]


@pytest.mark.parametrize(("retry_after", "delay"), [("6", 6.0), (None, 0.5)])
def test_terminal_rate_limit_still_defers_scope(
    clock: MockClock,
    monkeypatch: pytest.MonkeyPatch,
    retry_after: str | None,
    delay: float,
) -> None:
    monkeypatch.setattr(http.random, "uniform", lambda low, high: high / 2)
    quota = Quota()
    attempts: list[float] = []
    headers = {} if retry_after is None else {"retry-after": retry_after}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(clock.now)
        return httpx.Response(429, headers=headers)

    async def main() -> tuple[httpx.Response, httpx.Response]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            http_client = HTTPClient(client)
            first = await http_client.request(
                "POST", "https://api.test/write", quota=quota
            )
            second = await http_client.request(
                "GET", "https://api.test/read", quota=quota, replay_safe=False
            )
            return first, second

    first, second = asyncio.run(main())
    assert first.status_code == second.status_code == 429
    assert len(attempts) == 2
    assert attempts[1] - attempts[0] == pytest.approx(delay)
    assert clock.sleeps == [delay]


def test_cooldown_imposed_while_waiting_is_respected() -> None:
    quota = Quota(min_interval=0.15)
    starts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        starts.append(asyncio.get_running_loop().time())
        return httpx.Response(200)

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            http_client = HTTPClient(client)
            await http_client.request("GET", "https://api.test/first", quota=quota)
            waiter = asyncio.create_task(
                http_client.request("GET", "https://api.test/waiter", quota=quota)
            )
            await _REAL_SLEEP(0.01)
            # A rate limit on another request of the same scope.
            quota._defer(0.3)
            await waiter

    asyncio.run(main())
    assert starts[1] - starts[0] >= 0.28


def test_shared_quota_spaces_concurrent_attempts_including_retries(
    clock: MockClock, jitter: list[float]
) -> None:
    quota = Quota(min_interval=1.0)
    starts: list[float] = []
    served: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        starts.append(clock.now)
        path = request.url.path
        served[path] = served.get(path, 0) + 1
        return httpx.Response(503 if served[path] == 1 else 200)

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            http_client = HTTPClient(client)
            await asyncio.gather(
                *(
                    http_client.request("GET", f"https://api.test/{name}", quota=quota)
                    for name in "abc"
                )
            )

    asyncio.run(main())
    assert len(starts) == 6
    gaps = [later - earlier for earlier, later in itertools.pairwise(sorted(starts))]
    assert gaps == pytest.approx([1.0] * 5)


def test_independent_quotas_do_not_pace_each_other(
    clock: MockClock, jitter: list[float]
) -> None:
    starts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        starts.append(clock.now)
        return httpx.Response(200)

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            http_client = HTTPClient(client)
            quotas = [Quota(min_interval=5.0), Quota(min_interval=5.0)]
            await asyncio.gather(
                *(
                    http_client.request(
                        "GET", f"https://{index}.test/", quota=quotas[index]
                    )
                    for index in range(2)
                )
            )

    asyncio.run(main())
    assert starts == [1000.0, 1000.0]


@pytest.mark.parametrize("delay", ["30", "9" * 400], ids=["finite", "huge"])
def test_deadline_bounds_retry_after_wait(delay: str) -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(503, headers={"retry-after": delay})

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await HTTPClient(client, total_timeout=0.05).request(
                "GET", "https://api.test/"
            )

    with pytest.raises(TimeoutError):
        asyncio.run(main())
    assert len(attempts) == 1


@pytest.mark.parametrize("delay", ["30", "9" * 5000], ids=["finite", "huge"])
def test_deadline_bounds_initial_quota_wait(delay: str) -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(429, headers={"retry-after": delay})

    async def main() -> None:
        quota = Quota()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await HTTPClient(client, max_attempts=1).request(
                "GET", "https://api.test/rate-limited", quota=quota
            )
            await HTTPClient(client, total_timeout=0.05).request(
                "GET", "https://api.test/blocked", quota=quota
            )

    with pytest.raises(TimeoutError):
        asyncio.run(main())
    assert len(attempts) == 1


def test_timeout_covers_buffered_body_read() -> None:
    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            await _REAL_SLEEP(1.0)
            yield b"late"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=SlowBody())

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await HTTPClient(client, total_timeout=0.05).request(
                "GET", "https://api.test/"
            )

    with pytest.raises(TimeoutError):
        asyncio.run(main())


def test_cancellation_during_backoff_is_not_replayed() -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(503, headers={"retry-after": "30"})

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            task = asyncio.create_task(
                HTTPClient(client).request("GET", "https://api.test/")
            )
            while not attempts:
                await _REAL_SLEEP(0.001)
            await _REAL_SLEEP(0.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await _REAL_SLEEP(0.02)
            assert len(attempts) == 1

    asyncio.run(main())


def test_cancellation_during_quota_wait_keeps_the_gate() -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(429, headers={"retry-after": "1"})

    async def main() -> None:
        quota = Quota()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await HTTPClient(client, max_attempts=1).request(
                "GET", "https://api.test/rate-limited", quota=quota
            )
            waiter = asyncio.create_task(
                HTTPClient(client, total_timeout=10.0).request(
                    "GET", "https://api.test/waiter", quota=quota
                )
            )
            await _REAL_SLEEP(0.02)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            with pytest.raises(TimeoutError):
                await HTTPClient(client, total_timeout=0.05).request(
                    "GET", "https://api.test/blocked", quota=quota
                )

    asyncio.run(main())
    assert len(attempts) == 1


def test_redirects_are_not_followed(jitter: list[float]) -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(302, headers={"location": "https://elsewhere.test/"})

    async def main() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPClient(client).request("GET", "https://api.test/start")

    response = asyncio.run(main())
    assert response.status_code == 302
    assert response.headers["location"] == "https://elsewhere.test/"
    assert str(response.request.url) == "https://api.test/start"
    assert len(attempts) == 1


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_timeout": 0.0},
        {"total_timeout": -1.0},
        {"total_timeout": math.inf},
        {"total_timeout": math.nan},
        {"max_attempts": 0},
        {"max_attempts": -2},
    ],
)
def test_invalid_client_policy_is_rejected(kwargs: dict[str, float]) -> None:
    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_ok)) as client:
            with pytest.raises(ValueError):
                HTTPClient(client, **kwargs)

    asyncio.run(main())


@pytest.mark.parametrize("min_interval", [-0.1, math.inf, math.nan])
def test_invalid_quota_interval_is_rejected(min_interval: float) -> None:
    with pytest.raises(ValueError):
        Quota(min_interval)
