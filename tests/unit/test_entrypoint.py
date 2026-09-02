from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestMain:
    def test_exits_when_rq_is_unavailable(self) -> None:
        from src.worker import entrypoint

        with (
            patch.object(entrypoint, "_RQ_AVAILABLE", False),
            patch.object(entrypoint.log, "error"),
            patch.object(entrypoint.sys, "exit", side_effect=SystemExit(1)),
            pytest.raises(SystemExit, match="1"),
        ):
            entrypoint.main()

    def test_starts_worker_and_handles_sigterm_and_keyboard_interrupt(self) -> None:
        from src.worker import entrypoint

        worker = MagicMock()
        worker.work.side_effect = KeyboardInterrupt
        settings = SimpleNamespace(
            log_level="INFO",
            log_format="json",
            redis_job_timeout=120,
        )

        queue_mock = MagicMock()
        with (
            patch.object(entrypoint, "_RQ_AVAILABLE", True),
            patch.object(entrypoint, "get_settings", return_value=settings),
            patch.object(entrypoint, "configure_logging"),
            patch.object(entrypoint, "get_sync_redis", return_value="redis-conn"),
            patch.object(entrypoint, "Worker", return_value=worker) as mock_worker,
            patch.object(entrypoint, "Queue", queue_mock),
            patch.object(entrypoint.os, "getpid", return_value=999),
            patch.object(entrypoint.signal, "signal") as mock_signal,
        ):
            entrypoint.main()

        mock_worker.assert_called_once()
        worker.work.assert_called_once_with(with_scheduler=False)
        sigterm_handler = mock_signal.call_args.args[1]
        sigterm_handler(15, object())
        worker.request_stop.assert_called_once()

    def test_module_executes_main_when_run_as_script(self) -> None:
        worker = MagicMock()
        worker_factory = MagicMock(return_value=worker)
        fake_rq = ModuleType("rq")
        fake_rq.Worker = worker_factory
        fake_rq.Queue = MagicMock()
        fake_timeouts = ModuleType("rq.timeouts")
        fake_timeouts.JobTimeoutException = TimeoutError
        settings = SimpleNamespace(
            log_level="INFO",
            log_format="json",
            redis_job_timeout=120,
        )
        entrypoint_path = (
            Path(__file__).resolve().parents[2] / "src" / "worker" / "entrypoint.py"
        )

        with (
            patch.dict(sys.modules, {"rq": fake_rq, "rq.timeouts": fake_timeouts}),
            patch("src.config.get_settings", return_value=settings),
            patch("src.observability.tracing.configure_logging"),
            patch("src.worker.queue.get_sync_redis", return_value="redis-conn"),
            patch("signal.signal"),
            patch("os.getpid", return_value=321),
        ):
            runpy.run_path(str(entrypoint_path), run_name="__main__")

        worker_factory.assert_called_once()
        worker.work.assert_called_once_with(with_scheduler=False)


class TestHandleJobException:
    def test_logs_timeouts_and_returns_false(self) -> None:
        from src.worker import entrypoint

        class TimeoutTestError(Exception):
            pass

        logger = MagicMock()
        job = SimpleNamespace(id="job-123")

        with (
            patch.object(entrypoint, "JobTimeoutException", TimeoutTestError),
            patch.object(entrypoint, "log", logger),
        ):
            result = entrypoint._handle_job_exception(
                job,
                TimeoutTestError,
                TimeoutTestError("too slow"),
                object(),
            )

        assert result is False
        logger.error.assert_called_once()
        logger.warning.assert_called_once()

    def test_logs_non_timeout_failures_without_warning(self) -> None:
        from src.worker import entrypoint

        logger = MagicMock()
        job = SimpleNamespace(id="job-456")

        class TimeoutTestError(Exception):
            pass

        with (
            patch.object(entrypoint, "JobTimeoutException", TimeoutTestError),
            patch.object(entrypoint, "log", logger),
        ):
            result = entrypoint._handle_job_exception(
                job,
                RuntimeError,
                RuntimeError("boom"),
                object(),
            )

        assert result is False
        logger.error.assert_called_once()
        logger.warning.assert_not_called()
