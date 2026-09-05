"""
src/api/rate_limit.py — Token-bucket rate limiter using Redis.
Applied per GitHub installation ID to prevent runaway webhook floods
and protect against accidentally burning API quota.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

import structlog
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

import src.worker.queue as queue_module
from src.worker.queue import get_redis_connection

log = structlog.get_logger(__name__)


class RateLimiter:
    """
    Sliding-window rate limiter backed by Redis sorted sets.
    Each key (e.g., repo or IP) gets an independent bucket.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        key_prefix: str = "rate:",
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix

    async def check(self, identifier: str) -> tuple[bool, int]:
        """
        Check if identifier is within rate limit.
        Returns (allowed: bool, requests_remaining: int).
        """
        try:
            redis = await queue_module.get_redis_connection()
            key = f"{self.key_prefix}{identifier}"
            now = time.time()
            window_start = now - self.window_seconds
            entry_id = f"{now}:{uuid.uuid4().hex}"

            pipe = redis.pipeline()
            if hasattr(pipe, "__await__"):
                pipe = await pipe

            zrem = pipe.zremrangebyscore(key, 0, window_start)
            if hasattr(zrem, "__await__"):
                await zrem

            zc = pipe.zcard(key)
            if hasattr(zc, "__await__"):
                await zc

            za = pipe.zadd(key, {entry_id: now})
            if hasattr(za, "__await__"):
                await za

            exp = pipe.expire(key, self.window_seconds * 2)
            if hasattr(exp, "__await__"):
                await exp

            exec_res = pipe.execute()
            if hasattr(exec_res, "__await__"):
                results = await exec_res
            else:
                results = exec_res  # type: ignore[assignment]

            try:
                current_count = int(results[1])
            except (ValueError, TypeError, IndexError):
                current_count = 0

            allowed = current_count < self.max_requests
            remaining = max(0, self.max_requests - current_count - 1)

            if not allowed:
                log.warning(
                    "rate_limit.exceeded",
                    identifier=identifier,
                    count=current_count,
                    max=self.max_requests,
                    window_seconds=self.window_seconds,
                )

            return allowed, remaining
        except Exception as e:
            log.warning("rate_limit.backend_unavailable", error=str(e))
            return True, self.max_requests


# ── Pre-configured limiters ───────────────────────────────────────────────────

# Webhook endpoint: max 60 webhook events per repo per hour
# (a normal busy repo won't hit this; a misconfigured webhook loop will)
WEBHOOK_LIMITER = RateLimiter(
    max_requests=60,
    window_seconds=3600,
    key_prefix="webhook:",
)

# API trigger endpoint: max 10 manual triggers per repo per hour
API_TRIGGER_LIMITER = RateLimiter(
    max_requests=10,
    window_seconds=3600,
    key_prefix="api_trigger:",
)


# ── FastAPI dependency ────────────────────────────────────────────────────────


def make_rate_limit_dependency(limiter: RateLimiter, identifier_fn: Callable) -> Callable:
    """
    Factory for a FastAPI dependency that applies a rate limit.

    Usage:
        @router.post("/webhooks/github",
                     dependencies=[Depends(make_rate_limit_dependency(
                         WEBHOOK_LIMITER,
                         lambda req: req.headers.get("X-GitHub-Installation", "default")
                     ))])
    """

    async def _dependency(request: Request) -> None:
        identifier = identifier_fn(request)
        allowed, remaining = await limiter.check(identifier)

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"Max {limiter.max_requests} requests per {limiter.window_seconds}s window",
                    "retry_after": limiter.window_seconds,
                },
                headers={"Retry-After": str(limiter.window_seconds)},
            )

    return _dependency
