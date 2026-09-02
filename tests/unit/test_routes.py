from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException


class TestTriggerRun:
    @pytest.mark.asyncio
    async def test_rejects_manual_trigger_when_rate_limited(self) -> None:
        from src.api.routes import TriggerRequest, trigger_run

        with (
            patch(
                "src.api.routes.API_TRIGGER_LIMITER.check",
                new=AsyncMock(return_value=(False, 0)),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await trigger_run(
                TriggerRequest(repo_full_name="owner/repo", issue_number=42)
            )

        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_returns_existing_active_run(self) -> None:
        from src.api.routes import TriggerRequest, trigger_run

        repo = AsyncMock()
        repo.get_active_run.return_value = SimpleNamespace(
            run_id="run-existing",
            status="pending",
        )

        with (
            patch(
                "src.api.routes.API_TRIGGER_LIMITER.check",
                new=AsyncMock(return_value=(True, 1)),
            ),
            patch("src.api.routes.RunRepository", return_value=repo),
        ):
            result = await trigger_run(
                TriggerRequest(repo_full_name="owner/repo", issue_number=42)
            )

        assert result == {"job_id": "run-existing", "status": "pending"}

    @pytest.mark.asyncio
    async def test_enqueues_new_run(self) -> None:
        from src.api.routes import TriggerRequest, trigger_run

        repo = AsyncMock()
        repo.get_active_run.return_value = None

        with (
            patch(
                "src.api.routes.API_TRIGGER_LIMITER.check",
                new=AsyncMock(return_value=(True, 1)),
            ),
            patch("src.api.routes.RunRepository", return_value=repo),
            patch(
                "src.api.routes.enqueue_issue_job",
                new=AsyncMock(return_value="run-new"),
            ),
        ):
            result = await trigger_run(
                TriggerRequest(repo_full_name="owner/repo", issue_number=42)
            )

        assert result == {"job_id": "run-new", "status": "queued"}


class TestGetRun:
    @pytest.mark.asyncio
    async def test_raises_not_found_when_run_does_not_exist(self) -> None:
        from src.api.routes import get_run

        repo = AsyncMock()
        repo.get_run.return_value = None

        with (
            patch("src.api.routes.RunRepository", return_value=repo),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_run("missing")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_serialized_run(self) -> None:
        from src.api.routes import get_run

        run = SimpleNamespace(
            run_id="run-123",
            status="succeeded",
            issue_number=42,
            repo_full_name="owner/repo",
            pr_url="https://github.com/owner/repo/pull/1",
            retry_count=1,
            started_at=datetime(2026, 3, 29, tzinfo=UTC),
            completed_at=datetime(2026, 3, 29, 0, 1, tzinfo=UTC),
            failure_reason=None,
        )
        repo = AsyncMock()
        repo.get_run.return_value = run

        with patch("src.api.routes.RunRepository", return_value=repo):
            response = await get_run("run-123")

        assert response.run_id == "run-123"
        assert response.started_at == "2026-03-29T00:00:00+00:00"


class TestListRuns:
    @pytest.mark.asyncio
    async def test_lists_runs_with_limit_cap(self) -> None:
        from src.api.routes import list_runs

        run = SimpleNamespace(
            run_id="run-123",
            status="failed",
            issue_number=42,
            repo_full_name="owner/repo",
            pr_url=None,
            retry_count=2,
            started_at=None,
            completed_at=None,
        )
        repo = AsyncMock()
        repo.list_runs.return_value = [run]

        with patch("src.api.routes.RunRepository", return_value=repo):
            response = await list_runs(repo="owner/repo", status="failed", limit=500)

        repo.list_runs.assert_awaited_once_with(
            repo_full_name="owner/repo",
            status="failed",
            limit=100,
        )
        assert response[0].run_id == "run-123"
