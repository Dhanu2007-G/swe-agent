from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_checkpointer() -> None:
    aio_module = ModuleType("langgraph.checkpoint.postgres.aio")
    aio_module.AsyncPostgresSaver = MagicMock()
    monkey_modules = {
        "langgraph": ModuleType("langgraph"),
        "langgraph.checkpoint": ModuleType("langgraph.checkpoint"),
        "langgraph.checkpoint.postgres": ModuleType("langgraph.checkpoint.postgres"),
        "langgraph.checkpoint.postgres.aio": aio_module,
    }
    original_modules = {name: sys.modules.get(name) for name in monkey_modules}

    for name, module in monkey_modules.items():
        sys.modules[name] = module

    sys.modules.pop("src.agent.checkpointing", None)
    checkpointing = importlib.import_module("src.agent.checkpointing")

    checkpointing._checkpointer = None
    yield
    checkpointing._checkpointer = None
    sys.modules.pop("src.agent.checkpointing", None)

    for name, module in original_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class TestGetCheckpointer:
    @pytest.mark.asyncio
    async def test_initializes_and_reuses_shared_instance(self) -> None:
        from src.agent.checkpointing import get_checkpointer

        saver = AsyncMock()
        settings = SimpleNamespace(
            database_url="postgresql+asyncpg://user:pass@db/app",
        )

        with (
            patch("src.agent.checkpointing.get_settings", return_value=settings),
            patch(
                "src.agent.checkpointing.AsyncPostgresSaver.from_conn_string",
                return_value=saver,
            ) as mock_factory,
        ):
            first = await get_checkpointer()
            second = await get_checkpointer()

        assert first is saver
        assert second is saver
        mock_factory.assert_called_once_with("postgresql://user:pass@db/app")
        saver.setup.assert_awaited_once()


class TestCheckpointedRun:
    @pytest.mark.asyncio
    async def test_yields_checkpointer_and_logs_success(self) -> None:
        from src.agent.checkpointing import checkpointed_run

        saver = AsyncMock()
        logger = MagicMock()

        with (
            patch("src.agent.checkpointing.get_checkpointer", return_value=saver),
            patch("src.agent.checkpointing.log", logger),
        ):
            async with checkpointed_run("run-123") as result:
                assert result is saver

        logger.info.assert_any_call("checkpointer.run_start", run_id="run-123")
        logger.info.assert_any_call("checkpointer.run_complete", run_id="run-123")

    @pytest.mark.asyncio
    async def test_logs_and_reraises_run_errors(self) -> None:
        from src.agent.checkpointing import checkpointed_run

        saver = AsyncMock()
        logger = MagicMock()

        with (
            patch("src.agent.checkpointing.get_checkpointer", return_value=saver),
            patch("src.agent.checkpointing.log", logger),
            pytest.raises(RuntimeError, match="boom"),
        ):
            async with checkpointed_run("run-123"):
                raise RuntimeError("boom")

        logger.error.assert_called_once()


class TestRunState:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_checkpoint_exists(self) -> None:
        from src.agent.checkpointing import get_run_state

        saver = AsyncMock()
        saver.aget.return_value = None

        with patch("src.agent.checkpointing.get_checkpointer", return_value=saver):
            result = await get_run_state("run-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_channel_values_from_checkpoint(self) -> None:
        from src.agent.checkpointing import get_run_state

        saver = AsyncMock()
        saver.aget.return_value = {"channel_values": {"status": "running"}}

        with patch("src.agent.checkpointing.get_checkpointer", return_value=saver):
            result = await get_run_state("run-123")

        assert result == {"status": "running"}


class TestListCheckpointedRuns:
    @pytest.mark.asyncio
    async def test_lists_recent_runs_up_to_limit(self) -> None:
        from src.agent.checkpointing import list_checkpointed_runs

        class FakeSaver:
            def __init__(self) -> None:
                self.entries = [
                    SimpleNamespace(
                        metadata={"created_at": "2026-03-29T10:00:00Z"},
                        config={"configurable": {"thread_id": "run-1"}},
                        checkpoint={"id": "step-1"},
                    ),
                    SimpleNamespace(
                        metadata=None,
                        config=None,
                        checkpoint={"id": "step-2"},
                    ),
                ]

            async def alist(self, _: dict) -> object:
                for entry in self.entries:
                    yield entry

        with patch(
            "src.agent.checkpointing.get_checkpointer",
            return_value=FakeSaver(),
        ):
            runs = await list_checkpointed_runs(limit=1)

        assert runs == [
            {
                "thread_id": "run-1",
                "step": "step-1",
                "created_at": "2026-03-29T10:00:00Z",
            }
        ]
