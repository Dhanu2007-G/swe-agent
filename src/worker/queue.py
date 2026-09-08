"""
src/worker/queue.py — Redis queue for async job dispatch.
Uses RQ (Redis Queue) for reliable job processing with retry.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

import redis.asyncio as aioredis
import structlog
from redis import Redis as SyncRedis
from rq import Queue
from rq.job import Job
from sqlalchemy.exc import IntegrityError

from src.config import get_settings
from src.db.repository import RunRepository

log = structlog.get_logger(__name__)

QUEUE_NAME = "swe-agent-jobs"
HIGH_PRIORITY_QUEUE = "swe-agent-high"


def _active_job_key(repo_full_name: str, issue_number: int) -> str:
    """Format redis lock key for an active issue job."""
    return f"active-job:{repo_full_name}:{issue_number}"


async def release_active_job_lock(repo_full_name: str, issue_number: int, run_id: str) -> None:
    """Release the active job lock if it belongs to this run."""
    redis = await get_redis_connection()
    key = _active_job_key(repo_full_name, issue_number)
    current = await redis.get(key)
    if current == run_id:
        await redis.delete(key)


async def get_redis_connection() -> aioredis.Redis:
    """Return the async Redis connection. Lazily initialized."""
    settings = get_settings()
    return aioredis.from_url(
        str(settings.redis_url),
        encoding="utf-8",
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
        health_check_interval=30,
    )


def get_sync_redis() -> SyncRedis:
    """Sync Redis for RQ (RQ doesn't support async Redis yet)."""
    settings = get_settings()
    return SyncRedis.from_url(
        str(settings.redis_url),
        socket_timeout=5,
        socket_connect_timeout=5,
        decode_responses=False,
    )


async def enqueue_issue_job(
    repo_full_name: str,
    issue_number: int,
    delivery_id: str,
    high_priority: bool = False,
) -> str:
    """
    Enqueue an agent run job. Returns the job ID.
    Uses an active-job Redis lock and persists pending run in DB.
    """
    settings = get_settings()
    redis = await get_redis_connection()
    lock_key = _active_job_key(repo_full_name, issue_number)
    raw_id = str(uuid.uuid4().hex)
    job_id = f"run-{raw_id}"

    # Acquire lock
    acquired = await redis.set(lock_key, job_id, nx=True, ex=3600)
    if not acquired:
        existing_id = await redis.get(lock_key)
        return str(existing_id) if existing_id else job_id

    repo = RunRepository()
    try:
        await repo.create_run(
            run_id=job_id,
            repo_full_name=repo_full_name,
            issue_number=issue_number,
            status="pending",
        )
    except IntegrityError:
        await redis.delete(lock_key)
        active_run = await repo.get_active_run(repo_full_name, issue_number)
        if active_run:
            return active_run.run_id
        await repo.update_run(run_id=job_id, status="failed", failure_reason="DB conflict")
        job_key = f"job:{job_id}"
        await redis.hset(job_key, mapping={"status": "failed"})
        raise

    # Store job metadata in Redis for async polling
    job_key = f"job:{job_id}"
    await redis.hset(job_key, mapping={
        "job_id": job_id,
        "repo": repo_full_name,
        "issue_number": str(issue_number),
        "delivery_id": delivery_id,
        "status": "queued",
    })
    await redis.expire(job_key, settings.redis_result_ttl)

    # Enqueue via RQ (blocking call — run in executor)
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None,
            _enqueue_sync,
            job_id, repo_full_name, issue_number, high_priority, settings.redis_job_timeout,
        )
    except Exception as e:
        await repo.update_run(run_id=job_id, status="failed", failure_reason=str(e))
        await redis.hset(job_key, mapping={"status": "failed"})
        await redis.delete(lock_key)
        raise

    log.info("queue.job_enqueued", job_id=job_id, repo=repo_full_name,
             issue=issue_number, priority="high" if high_priority else "normal")
    return job_id


def _enqueue_sync(
    job_id: str,
    repo_full_name: str,
    issue_number: int,
    high_priority: bool,
    timeout: int,
) -> None:
    """Synchronous RQ enqueue — called from executor."""
    import rq
    from rq import Queue

    redis_conn = get_sync_redis()
    queue_name = HIGH_PRIORITY_QUEUE if high_priority else QUEUE_NAME
    q = Queue(queue_name, connection=redis_conn)

    retry_cls = getattr(rq, "Retry", None)
    retry_obj = retry_cls(max=3) if retry_cls else 3

    q.enqueue(
        "src.worker.processor.process_issue_job",
        kwargs={
            "repo_full_name": repo_full_name,
            "issue_number": issue_number,
            "run_id": job_id,
        },
        job_id=job_id,
        job_timeout=timeout,
        retry=retry_obj,
    )


async def get_job_status(job_id: str) -> dict[str, Any] | None:
    """Poll job status from Redis."""
    redis = await get_redis_connection()
    data = await redis.hgetall(f"job:{job_id}")
    return dict(data) if data else None
