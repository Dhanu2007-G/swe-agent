from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@contextmanager
def preserved_logging_state() -> object:
    root_logger = logging.getLogger()
    noisy_names = ["httpx", "httpcore", "uvicorn.access", "docker"]
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    original_noisy = {name: logging.getLogger(name).level for name in noisy_names}

    try:
        yield root_logger
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
        for name, level in original_noisy.items():
            logging.getLogger(name).setLevel(level)


def install_fake_otel(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, MagicMock]:
    trace_module = ModuleType("opentelemetry.trace")
    trace_module.set_tracer_provider = MagicMock()
    trace_module.get_tracer = MagicMock(return_value="tracer")

    otel_module = ModuleType("opentelemetry")
    otel_module.trace = trace_module

    exporter_module = ModuleType(
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    )
    exporter_module.OTLPSpanExporter = MagicMock(return_value="exporter")

    resource_module = ModuleType("opentelemetry.sdk.resources")

    class FakeResource:
        @staticmethod
        def create(data: dict[str, object]) -> dict[str, object]:
            return data

    resource_module.Resource = FakeResource

    trace_sdk_module = ModuleType("opentelemetry.sdk.trace")

    class FakeTracerProvider:
        def __init__(self, resource: dict[str, object]) -> None:
            self.resource = resource
            self.add_span_processor = MagicMock()

    trace_sdk_module.TracerProvider = FakeTracerProvider

    export_module = ModuleType("opentelemetry.sdk.trace.export")
    export_module.BatchSpanProcessor = MagicMock(return_value="processor")

    for name, module in {
        "opentelemetry": otel_module,
        "opentelemetry.trace": trace_module,
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": exporter_module,
        "opentelemetry.sdk.resources": resource_module,
        "opentelemetry.sdk.trace": trace_sdk_module,
        "opentelemetry.sdk.trace.export": export_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    return trace_module, exporter_module.OTLPSpanExporter


class TestConfigureLogging:
    @pytest.mark.parametrize("fmt", ["json", "console"])
    def test_configures_root_logger_and_noisy_loggers(self, fmt: str) -> None:
        from src.observability.tracing import configure_logging

        with preserved_logging_state() as root_logger:
            configure_logging(level="debug", fmt=fmt)

            assert root_logger.level == logging.DEBUG
            assert len(root_logger.handlers) == 1
            for noisy_name in ["httpx", "httpcore", "uvicorn.access", "docker"]:
                assert logging.getLogger(noisy_name).level == logging.WARNING


class TestConfigureTracing:
    def test_returns_early_when_no_endpoint_is_configured(self) -> None:
        from src.observability.tracing import configure_tracing

        settings = SimpleNamespace(otel_exporter_otlp_endpoint=None)

        with patch("src.config.get_settings", return_value=settings):
            configure_tracing()

    def test_configures_otel_when_endpoint_is_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.observability.tracing import configure_tracing

        trace_module, mock_exporter_cls = install_fake_otel(monkeypatch)
        settings = SimpleNamespace(
            otel_exporter_otlp_endpoint="http://otel:4317",
            is_production=False,
            app_env="development",
        )
        logger = MagicMock()

        with (
            patch("src.config.get_settings", return_value=settings),
            patch("src.observability.tracing.structlog.get_logger", return_value=logger),
        ):
            configure_tracing()

        trace_module.set_tracer_provider.assert_called_once()
        mock_exporter_cls.assert_called_once_with(
            endpoint="http://otel:4317",
            insecure=True,
        )
        logger.info.assert_called_once()

    def test_swallows_import_errors(self) -> None:
        from src.observability.tracing import configure_tracing

        settings = SimpleNamespace(
            otel_exporter_otlp_endpoint="http://otel:4317",
            is_production=True,
            app_env="production",
        )
        original_import = __import__

        def fake_import(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name.startswith("opentelemetry"):
                raise ImportError("missing")
            return original_import(name, globals, locals, fromlist, level)

        with (
            patch("src.config.get_settings", return_value=settings),
            patch("builtins.__import__", side_effect=fake_import),
        ):
            configure_tracing()


class TestGetTracer:
    def test_returns_tracer_when_opentelemetry_is_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.observability.tracing import get_tracer

        trace_module, _ = install_fake_otel(monkeypatch)
        assert get_tracer("swe-agent") == "tracer"
        trace_module.get_tracer.assert_called_once_with("swe-agent")

    def test_returns_none_when_opentelemetry_is_missing(self) -> None:
        from src.observability.tracing import get_tracer

        original_import = __import__

        def fake_import(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "opentelemetry":
                raise ImportError("missing")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            assert get_tracer("swe-agent") is None


class TestMetrics:
    @pytest.mark.asyncio
    async def test_record_run_metrics_observes_all_metrics(self) -> None:
        from src.observability import tracing

        runs_total = MagicMock()
        retries = MagicMock()
        duration = MagicMock()
        tokens = MagicMock()
        settings = SimpleNamespace(anthropic_model="claude-opus-4-6")
        issue = SimpleNamespace(repo_full_name="owner/repo")

        with (
            patch.object(tracing, "AGENT_RUNS_TOTAL", runs_total),
            patch.object(tracing, "AGENT_RETRIES", retries),
            patch.object(tracing, "AGENT_RUN_DURATION", duration),
            patch.object(tracing, "AGENT_TOKENS_USED", tokens),
            patch("src.config.get_settings", return_value=settings),
        ):
            await tracing.record_run_metrics({
                "status": "succeeded",
                "issue": issue,
                "retry_count": 2,
                "started_at": "2026-03-29T10:00:00",
                "completed_at": "2026-03-29T10:00:05",
                "total_tokens_used": 1234,
            })

        runs_total.labels.return_value.inc.assert_called_once()
        retries.labels.return_value.observe.assert_called_once_with(2)
        duration.labels.return_value.observe.assert_called_once_with(5.0)
        tokens.labels.return_value.observe.assert_called_once_with(1234)

    @pytest.mark.asyncio
    async def test_record_run_metrics_ignores_invalid_duration(self) -> None:
        from src.observability import tracing

        duration = MagicMock()

        with patch.object(tracing, "AGENT_RUN_DURATION", duration):
            await tracing.record_run_metrics({
                "status": "failed",
                "issue": None,
                "retry_count": 0,
                "started_at": "not-a-date",
                "completed_at": "still-not-a-date",
            })

        duration.labels.return_value.observe.assert_not_called()

    def test_active_run_helpers_delegate_to_gauge(self) -> None:
        from src.observability import tracing

        gauge = MagicMock()

        with patch.object(tracing, "ACTIVE_RUNS", gauge):
            tracing.increment_active_runs()
            tracing.decrement_active_runs()

        gauge.inc.assert_called_once()
        gauge.dec.assert_called_once()
