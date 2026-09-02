from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db.database import AgentRun
from src.db.repository import RunRepository


def one_session(session: object) -> object:
    async def _generator() -> object:
        yield session

    return _generator()


def exhausted_sessions() -> object:
    async def _generator() -> object:
        if False:
            yield None

    return _generator()


class TestRunRepository:
    @pytest.mark.asyncio
    async def test_create_run_adds_and_flushes_model(self) -> None:
        session = AsyncMock()
        session.add = MagicMock()
        started_at = datetime.now(UTC)

        with patch("src.db.repository.get_session", lambda: one_session(session)):
            run = await RunRepository().create_run(
                run_id="run-123",
                repo_full_name="owner/repo",
                issue_number=42,
                status="pending",
                started_at=started_at,
            )

        assert isinstance(run, AgentRun)
        assert run.run_id == "run-123"
        assert run.status == "pending"
        assert run.started_at == started_at
        session.add.assert_called_once()
        session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_run_raises_when_session_generator_is_empty(self) -> None:
        with (
            patch("src.db.repository.get_session", exhausted_sessions),
            pytest.raises(RuntimeError, match="Session exhausted"),
        ):
            await RunRepository().create_run("run-1", "owner/repo", 1)

    @pytest.mark.asyncio
    async def test_mark_run_running_delegates_to_update_run(self) -> None:
        repo = RunRepository()

        with patch.object(repo, "update_run", new_callable=AsyncMock) as mock_update:
            await repo.mark_run_running("run-123")

        mock_update.assert_awaited_once()
        assert mock_update.await_args.kwargs["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_run_returns_scalar_result(self) -> None:
        run = SimpleNamespace(run_id="run-123")
        result = MagicMock()
        result.scalar_one_or_none.return_value = run
        session = AsyncMock()
        session.execute.return_value = result

        with patch("src.db.repository.get_session", lambda: one_session(session)):
            found = await RunRepository().get_run("run-123")

        assert found is run

    @pytest.mark.asyncio
    async def test_get_run_returns_none_when_session_is_empty(self) -> None:
        with patch("src.db.repository.get_session", exhausted_sessions):
            assert await RunRepository().get_run("run-123") is None

    @pytest.mark.asyncio
    async def test_get_active_run_returns_scalar_result(self) -> None:
        run = SimpleNamespace(run_id="run-active")
        result = MagicMock()
        result.scalar_one_or_none.return_value = run
        session = AsyncMock()
        session.execute.return_value = result

        with patch("src.db.repository.get_session", lambda: one_session(session)):
            found = await RunRepository().get_active_run("owner/repo", 42)

        assert found is run

    @pytest.mark.asyncio
    async def test_get_active_run_returns_none_when_session_is_empty(self) -> None:
        with patch("src.db.repository.get_session", exhausted_sessions):
            assert await RunRepository().get_active_run("owner/repo", 42) is None

    @pytest.mark.asyncio
    async def test_update_run_builds_values_for_optional_fields(self) -> None:
        session = AsyncMock()
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)

        with patch("src.db.repository.get_session", lambda: one_session(session)):
            await RunRepository().update_run(
                run_id="run-123",
                status="failed",
                pr_url="https://example.com/pr/1",
                retry_count=2,
                failure_reason="x" * 600,
                completed_at=completed_at,
                state_snapshot='{"status":"failed"}',
                started_at=started_at,
            )

        statement = session.execute.await_args.args[0]
        compiled = statement.compile()
        assert compiled.params["status"] == "failed"
        assert compiled.params["retry_count"] == 2
        assert compiled.params["pr_url"] == "https://example.com/pr/1"
        assert compiled.params["failure_reason"] == "x" * 500
        assert compiled.params["state_snapshot"] == '{"status":"failed"}'

    @pytest.mark.asyncio
    async def test_list_runs_returns_query_results_and_empty_fallback(self) -> None:
        run = SimpleNamespace(run_id="run-123")
        result = MagicMock()
        result.scalars.return_value.all.return_value = [run]
        session = AsyncMock()
        session.execute.return_value = result

        with patch("src.db.repository.get_session", lambda: one_session(session)):
            runs = await RunRepository().list_runs(
                repo_full_name="owner/repo",
                status="running",
                limit=10,
            )

        assert runs == [run]

        with patch("src.db.repository.get_session", exhausted_sessions):
            assert await RunRepository().list_runs() == []
