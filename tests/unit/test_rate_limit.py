from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.api.rate_limit import RateLimiter, make_rate_limit_dependency


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_allows_request_and_tracks_remaining_with_async_pipeline(self) -> None:
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        pipe = AsyncMock()
        pipe.execute.return_value = [0, 2, 1, True]
        redis = AsyncMock()
        redis.pipeline = AsyncMock(return_value=pipe)

        class FakeUUID:
            hex = "0" * 32

        with (
            patch(
                "src.api.rate_limit.queue_module.get_redis_connection",
                new_callable=AsyncMock,
                return_value=redis,
            ),
            patch("src.api.rate_limit.time.time", return_value=1_000.0),
            patch("src.api.rate_limit.uuid.uuid4", return_value=FakeUUID()),
        ):
            allowed, remaining = await limiter.check("owner/repo")

        assert allowed is True
        assert remaining == 2
        pipe.zremrangebyscore.assert_awaited_once_with("rate:owner/repo", 0, 940.0)
        pipe.zcard.assert_awaited_once_with("rate:owner/repo")
        pipe.zadd.assert_awaited_once_with("rate:owner/repo", {"1000.0:" + "0" * 32: 1000.0})
        pipe.expire.assert_awaited_once_with("rate:owner/repo", 120)

    @pytest.mark.asyncio
    async def test_rejects_request_when_limit_is_reached(self) -> None:
        limiter = RateLimiter(max_requests=3, window_seconds=60)

        class SyncPipeline:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def zremrangebyscore(self, key: str, minimum: int, maximum: float) -> None:
                self.calls.append(("zremrangebyscore", key, minimum, maximum))

            def zcard(self, key: str) -> None:
                self.calls.append(("zcard", key))

            def zadd(self, key: str, mapping: dict[str, float]) -> None:
                self.calls.append(("zadd", key, mapping))

            def expire(self, key: str, ttl: int) -> None:
                self.calls.append(("expire", key, ttl))

            def execute(self) -> list[int]:
                return [0, 3, 1, 1]

        pipe = SyncPipeline()
        redis = SimpleNamespace(pipeline=lambda: pipe)

        with (
            patch(
                "src.api.rate_limit.queue_module.get_redis_connection",
                new_callable=AsyncMock,
                return_value=redis,
            ),
            patch("src.api.rate_limit.log.warning") as warning_mock,
        ):
            allowed, remaining = await limiter.check("owner/repo")

        assert allowed is False
        assert remaining == 0
        warning_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_fails_open_when_backend_is_unavailable(self) -> None:
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        with patch(
            "src.api.rate_limit.queue_module.get_redis_connection",
            new_callable=AsyncMock,
            side_effect=RuntimeError("redis down"),
        ):
            allowed, remaining = await limiter.check("owner/repo")

        assert allowed is True
        assert remaining == 10

    @pytest.mark.asyncio
    async def test_handles_non_integer_pipeline_result(self) -> None:
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        pipe = AsyncMock()
        pipe.execute.return_value = [0, "invalid-count", 1, True]
        redis = AsyncMock()
        redis.pipeline = AsyncMock(return_value=pipe)

        with patch(
            "src.api.rate_limit.queue_module.get_redis_connection",
            new_callable=AsyncMock,
            return_value=redis,
        ):
            allowed, remaining = await limiter.check("owner/repo")

        assert allowed is True
        assert remaining == 4


class TestRateLimitDependency:
    @pytest.mark.asyncio
    async def test_allows_request_when_limiter_allows_it(self) -> None:
        limiter = AsyncMock()
        limiter.check.return_value = (True, 4)
        limiter.max_requests = 5
        limiter.window_seconds = 60
        dependency = make_rate_limit_dependency(
            limiter,
            lambda request: request.headers["x-repo"],
        )

        request = SimpleNamespace(headers={"x-repo": "owner/repo"})

        await dependency(request)

        limiter.check.assert_awaited_once_with("owner/repo")

    @pytest.mark.asyncio
    async def test_raises_http_429_when_rate_limit_is_exceeded(self) -> None:
        limiter = AsyncMock()
        limiter.check.return_value = (False, 0)
        limiter.max_requests = 5
        limiter.window_seconds = 60
        dependency = make_rate_limit_dependency(
            limiter,
            lambda request: request.headers["x-repo"],
        )

        request = SimpleNamespace(headers={"x-repo": "owner/repo"})

        with pytest.raises(HTTPException) as exc_info:
            await dependency(request)

        assert exc_info.value.status_code == 429
        assert exc_info.value.headers == {"Retry-After": "60"}
        assert exc_info.value.detail == {
            "error": "rate_limit_exceeded",
            "message": "Max 5 requests per 60s window",
            "retry_after": 60,
        }
