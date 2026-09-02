from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class FakeSessionContext:
    def __init__(self, session: object) -> None:
        self.session = session

    async def __aenter__(self) -> object:
        return self.session

    async def __aexit__(self, *_: object) -> None:
        return None


class FakeBeginContext:
    def __init__(self, conn: object) -> None:
        self.conn = conn

    async def __aenter__(self) -> object:
        return self.conn

    async def __aexit__(self, *_: object) -> None:
        return None


class TestInitDb:
    @pytest.mark.asyncio
    async def test_initializes_engine_sessionmaker_and_creates_tables(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.db import database

        conn = AsyncMock()
        engine = MagicMock()
        engine.begin.return_value = FakeBeginContext(conn)
        session_factory = MagicMock()
        settings = SimpleNamespace(
            database_url="postgresql+asyncpg://user:pass@db/app",
            database_pool_size=5,
            database_max_overflow=10,
            database_echo=False,
        )

        monkeypatch.setattr(database, "_engine", None)
        monkeypatch.setattr(database, "_session_factory", None)

        with (
            patch("src.db.database.get_settings", return_value=settings),
            patch("src.db.database.create_async_engine", return_value=engine),
            patch("src.db.database.async_sessionmaker", return_value=session_factory),
        ):
            await database.init_db()

        assert database._engine is engine
        assert database._session_factory is session_factory
        conn.run_sync.assert_awaited_once_with(database.Base.metadata.create_all)


class TestGetSession:
    @pytest.mark.asyncio
    async def test_raises_when_database_is_not_initialized(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.db import database

        monkeypatch.setattr(database, "_session_factory", None)

        with pytest.raises(RuntimeError, match="Database not initialized"):
            await anext(database.get_session())

    @pytest.mark.asyncio
    async def test_commits_after_successful_use(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.db import database

        session = AsyncMock()
        monkeypatch.setattr(
            database,
            "_session_factory",
            lambda: FakeSessionContext(session),
        )

        session_iter = database.get_session()
        yielded = await anext(session_iter)

        assert yielded is session

        with pytest.raises(StopAsyncIteration):
            await anext(session_iter)

        session.commit.assert_awaited_once()
        session.rollback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rolls_back_when_consumer_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.db import database

        session = AsyncMock()
        monkeypatch.setattr(
            database,
            "_session_factory",
            lambda: FakeSessionContext(session),
        )

        session_iter = database.get_session()
        await anext(session_iter)

        with pytest.raises(RuntimeError, match="boom"):
            await session_iter.athrow(RuntimeError("boom"))

        session.rollback.assert_awaited_once()


class TestAgentRunModel:
    def test_repr_includes_run_id_issue_and_status(self) -> None:
        from src.db.database import AgentRun

        run = AgentRun(
            run_id="run-123",
            repo_full_name="owner/repo",
            issue_number=42,
            status="running",
        )

        assert repr(run) == "<AgentRun run_id='run-123' issue=42 status='running'>"
