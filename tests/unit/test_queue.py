from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.exc import IntegrityError


class TestQueueHelpers:
    @pytest.mark.asyncio
    async def test_get_redis_connection_uses_settings(self) -> None:
        from src.worker.queue import get_redis_connection

        settings = SimpleNamespace(redis_url="redis://redis:6379/0")

        with (
            patch("src.worker.queue.get_settings", return_value=settings),
            patch("src.worker.queue.aioredis.from_url", return_value="async-redis") as from_url,
        ):
            redis = await get_redis_connection()

        assert redis == "async-redis"
        from_url.assert_called_once_with(
            "redis://redis:6379/0",
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            health_check_interval=30,
        )

    def test_get_sync_redis_uses_settings(self) -> None:
        from src.worker.queue import get_sync_redis

        settings = SimpleNamespace(redis_url="redis://redis:6379/0")

        with (
            patch("src.worker.queue.get_settings", return_value=settings),
            patch("src.worker.queue.SyncRedis.from_url", return_value="sync-redis") as from_url,
        ):
            redis = get_sync_redis()

        assert redis == "sync-redis"
        from_url.assert_called_once_with(
            "redis://redis:6379/0",
            socket_timeout=5,
            socket_connect_timeout=5,
            decode_responses=False,
        )

    def test_enqueue_sync_uses_expected_queue_and_retry_settings(self) -> None:
        from src.worker.queue import _enqueue_sync

        captured: dict[str, object] = {}

        class FakeRetry:
            def __init__(self, max: int, intervals: list[int] | None = None) -> None:
                self.max = max
                self.intervals = intervals or [10, 30, 60]

        class FakeQueue:
            def __init__(self, name: str, connection: str, **kwargs: object) -> None:
                captured["name"] = name
                captured["connection"] = connection
                captured.update(kwargs)

            def enqueue(self, *args: object, **kwargs: object) -> None:
                captured["args"] = args
                captured["kwargs"] = kwargs

        with (
            patch("src.worker.queue.get_sync_redis", return_value="sync-redis"),
            patch("src.worker.queue.Queue", FakeQueue),
            patch.dict(
                "sys.modules",
                {
                    "rq": SimpleNamespace(Queue=FakeQueue, Retry=FakeRetry),
                    "rq.serializers": SimpleNamespace(JSONSerializer=str),
                },
            ),
        ):
            _enqueue_sync("run-123", "owner/repo", 42, True, 900)

        kwargs = captured["kwargs"]
        assert captured["name"] == "swe-agent-high"
        assert captured["connection"] == "sync-redis"
        assert kwargs["job_id"] == "run-123"
        assert kwargs["kwargs"] == {
            "repo_full_name": "owner/repo",
            "issue_number": 42,
            "run_id": "run-123",
        }
        assert kwargs["job_timeout"] == 900
        assert kwargs["retry"].max == 3

    @pytest.mark.asyncio
    async def test_get_job_status_returns_dict_when_present(self) -> None:
        from src.worker.queue import get_job_status

        redis = AsyncMock()
        redis.hgetall.return_value = {"status": "queued", "job_id": "run-123"}

        with patch(
            "src.worker.queue.get_redis_connection", new_callable=AsyncMock, return_value=redis
        ):
            result = await get_job_status("run-123")

        assert result == {"status": "queued", "job_id": "run-123"}

    @pytest.mark.asyncio
    async def test_get_job_status_returns_none_when_missing(self) -> None:
        from src.worker.queue import get_job_status

        redis = AsyncMock()
        redis.hgetall.return_value = {}

        with patch(
            "src.worker.queue.get_redis_connection", new_callable=AsyncMock, return_value=redis
        ):
            result = await get_job_status("run-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_release_active_job_lock_noops_for_other_run(self) -> None:
        from src.worker.queue import release_active_job_lock

        redis = AsyncMock()
        redis.get.return_value = "run-other"

        with patch(
            "src.worker.queue.get_redis_connection", new_callable=AsyncMock, return_value=redis
        ):
            await release_active_job_lock("owner/repo", 42, "run-123")

        redis.delete.assert_not_awaited()

    def test_active_job_key_formats_repo_and_issue(self) -> None:
        from src.worker.queue import _active_job_key

        assert _active_job_key("owner/repo", 42) == "active-job:owner/repo:42"


class TestEnqueueIssueJob:
    @pytest.mark.asyncio
    async def test_returns_existing_job_when_lock_is_held(self) -> None:
        from src.worker.queue import enqueue_issue_job

        redis = AsyncMock()
        redis.set.return_value = False
        redis.get.return_value = "run-existing"
        repo = AsyncMock()

        with (
            patch(
                "src.worker.queue.get_redis_connection",
                new_callable=AsyncMock,
                return_value=redis,
            ),
            patch("src.worker.queue.RunRepository", return_value=repo),
        ):
            job_id = await enqueue_issue_job("owner/repo", 42, "delivery-1")

        assert job_id == "run-existing"
        redis.hset.assert_not_awaited()
        repo.create_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_releases_lock_when_enqueue_fails(self) -> None:
        from src.worker.queue import enqueue_issue_job

        redis = AsyncMock()
        redis.set.return_value = True
        redis.get.return_value = "run-0000000000000000"
        repo = AsyncMock()

        with (
            patch(
                "src.worker.queue.get_redis_connection",
                new_callable=AsyncMock,
                return_value=redis,
            ),
            patch("src.worker.queue._enqueue_sync", side_effect=RuntimeError("boom")),
            patch(
                "src.worker.queue.uuid.uuid4",
                return_value=SimpleNamespace(hex="0000000000000000"),
            ),
            patch("src.worker.queue.RunRepository", return_value=repo),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await enqueue_issue_job("owner/repo", 42, "delivery-1")

        repo.create_run.assert_awaited_once()
        repo.update_run.assert_awaited_once()
        redis.delete.assert_awaited_once()
        assert redis.hset.await_count == 2
        assert redis.hset.await_args_list[-1].kwargs["mapping"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_persists_pending_run_before_enqueue(self) -> None:
        from src.worker.queue import enqueue_issue_job

        redis = AsyncMock()
        redis.set.return_value = True
        repo = AsyncMock()

        with (
            patch(
                "src.worker.queue.get_redis_connection",
                new_callable=AsyncMock,
                return_value=redis,
            ),
            patch(
                "src.worker.queue.uuid.uuid4",
                return_value=SimpleNamespace(hex="1111111111111111"),
            ),
            patch("src.worker.queue.RunRepository", return_value=repo),
            patch("src.worker.queue._enqueue_sync"),
        ):
            job_id = await enqueue_issue_job("owner/repo", 42, "delivery-1")

        assert job_id == "run-1111111111111111"
        repo.create_run.assert_awaited_once_with(
            run_id="run-1111111111111111",
            repo_full_name="owner/repo",
            issue_number=42,
            status="pending",
        )

    @pytest.mark.asyncio
    async def test_returns_existing_run_when_db_unique_constraint_hits(self) -> None:
        from src.worker.queue import enqueue_issue_job

        redis = AsyncMock()
        redis.set.return_value = True
        redis.get.return_value = "run-2222222222222222"
        repo = AsyncMock()
        repo.create_run.side_effect = IntegrityError("dup", params=None, orig=None)
        repo.get_active_run.return_value = SimpleNamespace(run_id="run-existing-db")

        with (
            patch(
                "src.worker.queue.get_redis_connection",
                new_callable=AsyncMock,
                return_value=redis,
            ),
            patch(
                "src.worker.queue.uuid.uuid4",
                return_value=SimpleNamespace(hex="2222222222222222"),
            ),
            patch("src.worker.queue.RunRepository", return_value=repo),
            patch("src.worker.queue._enqueue_sync") as enqueue_sync,
        ):
            job_id = await enqueue_issue_job("owner/repo", 42, "delivery-1")

        assert job_id == "run-existing-db"
        enqueue_sync.assert_not_called()
        redis.hset.assert_not_awaited()
        redis.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_re_raises_integrity_error_when_active_run_lookup_is_empty(self) -> None:
        from src.worker.queue import enqueue_issue_job

        redis = AsyncMock()
        redis.set.return_value = True
        redis.get.return_value = "run-3333333333333333"
        repo = AsyncMock()
        repo.create_run.side_effect = IntegrityError("dup", params=None, orig=None)
        repo.get_active_run.return_value = None

        with (
            patch(
                "src.worker.queue.get_redis_connection",
                new_callable=AsyncMock,
                return_value=redis,
            ),
            patch(
                "src.worker.queue.uuid.uuid4",
                return_value=SimpleNamespace(hex="3333333333333333"),
            ),
            patch("src.worker.queue.RunRepository", return_value=repo),
            pytest.raises(IntegrityError),
        ):
            await enqueue_issue_job("owner/repo", 42, "delivery-1")

        repo.update_run.assert_awaited_once()
        redis.delete.assert_awaited_once()
        assert redis.hset.await_args_list[-1].kwargs["mapping"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_release_active_job_lock_matching(self) -> None:
        from src.worker.queue import release_active_job_lock

        redis = AsyncMock()
        redis.get.return_value = "run-123"
        with patch(
            "src.worker.queue.get_redis_connection", new_callable=AsyncMock, return_value=redis
        ):
            await release_active_job_lock("owner/repo", 42, "run-123")
        redis.delete.assert_awaited_once_with("active-job:owner/repo:42")
