"""
src/observability/tracing.py — Logging, tracing, and metrics configuration.
Configures structlog for JSON output, OTEL for distributed traces,
and Prometheus for custom agent metrics.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram

# ── Agent-specific Prometheus Metrics ─────────────────────────────────────────

AGENT_RUNS_TOTAL = Counter(
    "swe_agent_runs_total",
    "Total agent runs",
    ["status", "repo"],
)

AGENT_RUN_DURATION = Histogram(
    "swe_agent_run_duration_seconds",
    "Time from issue receipt to PR/fail",
    ["status"],
    buckets=[30, 60, 120, 300, 600, 1200, 1800],
)

AGENT_RETRIES = Histogram(
    "swe_agent_retries",
    "Number of self-correction retries per run",
    ["final_status"],
    buckets=[0, 1, 2, 3],
)

AGENT_TOKENS_USED = Histogram(
    "swe_agent_tokens_used",
    "LLM tokens consumed per run",
    ["model"],
    buckets=[1000, 5000, 10000, 25000, 50000, 100000, 200000],
)

ACTIVE_RUNS = Gauge(
    "swe_agent_active_runs",
    "Currently running agent jobs",
)

QUEUE_DEPTH = Gauge(
    "swe_agent_queue_depth",
    "Current number of queued jobs waiting for processing",
    ["queue_name"],
)

LLM_REQUEST_DURATION = Histogram(
    "swe_agent_llm_request_duration_seconds",
    "Duration of LLM API requests in seconds",
    ["model", "node"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

AGENT_RUN_COST_USD = Histogram(
    "swe_agent_run_cost_usd",
    "Cumulative financial cost of LLM calls per agent run in USD",
    ["status"],
    buckets=[0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00],
)

PR_CREATION_TOTAL = Counter(
    "swe_agent_pr_creation_total",
    "Total number of pull requests created or updated",
    ["status", "is_fork"],
)

SANDBOX_EXECUTION_TIME = Histogram(
    "swe_agent_sandbox_execution_seconds",
    "Docker sandbox test execution time",
    buckets=[5, 10, 30, 60, 120, 300],
)


# ── Logging Configuration ─────────────────────────────────────────────────────


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """
    Configure structlog for structured JSON output in production,
    or pretty console output in development.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)  # type: ignore[assignment]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *shared_processors,
            renderer,
        ]
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper()))

    # Quiet noisy third-party loggers
    for noisy in ["httpx", "httpcore", "uvicorn.access", "docker"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── OpenTelemetry ─────────────────────────────────────────────────────────────


def configure_tracing() -> None:
    """Configure OTEL tracing if endpoint is configured."""
    from src.config import get_settings

    settings = get_settings()

    if not settings.otel_exporter_otlp_endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": "swe-agent",
                "service.version": "1.0.0",
                "deployment.environment": settings.app_env,
            }
        )

        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            insecure=not settings.is_production,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        structlog.get_logger(__name__).info(
            "otel.configured",
            endpoint=settings.otel_exporter_otlp_endpoint,
        )
    except ImportError:
        pass  # OTEL packages not installed


def get_tracer(name: str) -> Any:
    """Get a tracer for manual instrumentation."""
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return None


# ── Metrics Recording ─────────────────────────────────────────────────────────


async def record_run_metrics(final_state: dict[str, Any]) -> None:
    """Record Prometheus metrics after a run completes."""
    from datetime import datetime

    status = final_state.get("status", "unknown")
    issue = final_state.get("issue")
    repo = issue.repo_full_name if issue else "unknown"

    AGENT_RUNS_TOTAL.labels(status=status, repo=repo).inc()

    retry_count = final_state.get("retry_count", 0)
    AGENT_RETRIES.labels(final_status=status).observe(retry_count)

    # Calculate duration
    started_at_str = final_state.get("started_at")
    completed_at_str = final_state.get("completed_at")
    if started_at_str and completed_at_str:
        try:
            start = datetime.fromisoformat(started_at_str)
            end = datetime.fromisoformat(completed_at_str)
            duration = (end - start).total_seconds()
            AGENT_RUN_DURATION.labels(status=status).observe(duration)
        except ValueError:
            pass

    tokens = final_state.get("total_tokens_used", 0)
    if tokens:
        from src.config import get_settings

        AGENT_TOKENS_USED.labels(model=get_settings().anthropic_model).observe(tokens)

    cost = final_state.get("cumulative_cost_usd", 0.0)
    if cost > 0:
        AGENT_RUN_COST_USD.labels(status=status).observe(cost)

    pr = final_state.get("pull_request")
    if pr:
        is_fork = str(final_state.get("is_fork", False)).lower()
        PR_CREATION_TOTAL.labels(status=status, is_fork=is_fork).inc()


def increment_active_runs() -> None:
    """Increment the active runs gauge."""
    ACTIVE_RUNS.inc()


def decrement_active_runs() -> None:
    """Decrement the active runs gauge."""
    ACTIVE_RUNS.dec()
