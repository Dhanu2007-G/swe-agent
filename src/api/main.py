"""
src/api/main.py — FastAPI application entrypoint.
Handles GitHub webhooks, job status API, and health endpoints.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.webhook import router as webhook_router
from src.api.routes import router as api_router
from src.api.security import RequestValidationMiddleware, SecurityHeadersMiddleware
from src.config import get_settings
from src.db.database import init_db
from src.observability.tracing import configure_logging, configure_tracing
from src.worker.queue import get_redis_connection

log = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
WEBHOOK_COUNTER = Counter(
    "github_webhooks_total",
    "GitHub webhooks received",
    ["event", "action"],
)


# ── Lifespan ──────────────────────────────────────────────────────────────────


async def close_redis_connection() -> None:
    """Close redis connection pool."""
    try:
        redis = await get_redis_connection()
        if hasattr(redis, "aclose"):
            await redis.aclose()
        elif hasattr(redis, "close"):
            await redis.close()
    except Exception:
        log.warning("app.redis_close_failed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown logic. Runs once per process."""
    settings = get_settings()

    # Configure logging before anything else
    configure_logging(getattr(settings, "log_level", "INFO"), getattr(settings, "log_format", "json"))
    configure_tracing()

    log.info("app.starting", env=getattr(settings, "app_env", "development"))

    # Verify dependencies are reachable (tolerant to offline redis in test/mock mode)
    try:
        redis = await get_redis_connection()
        await redis.ping()
        log.info("app.redis_ok")
    except Exception as e:
        log.warning("app.redis_connect_failed", error=str(e))

    try:
        await init_db()
        log.info("app.db_ok")
    except Exception as e:
        log.warning("app.db_init_failed", error=str(e))

    log.info("app.ready", env=getattr(settings, "app_env", "development"))

    yield

    log.info("app.shutting_down")
    try:
        await close_redis_connection()
    except Exception:
        log.warning("app.shutdown_redis_close_failed")


# ── Application Factory ───────────────────────────────────────────────────────


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="SWE Agent API",
        version="1.0.0",
        docs_url="/docs" if not getattr(settings, "is_production", False) else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    # ── Security Middleware ───────────────────────────────────────────────────
    if getattr(settings, "is_production", False):
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=getattr(settings, "allowed_hosts", ["*"]),
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not getattr(settings, "is_production", False) else [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Security headers + request validation (runs after CORS, before routes)
    app.add_middleware(
        SecurityHeadersMiddleware,
        is_production=getattr(settings, "is_production", False),
    )
    app.add_middleware(RequestValidationMiddleware)

    # ── Request logging + metrics middleware ──────────────────────────────────
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next: object) -> Response:
        start = time.monotonic()
        response: Response = await call_next(request)  # type: ignore[operator]
        duration = time.monotonic() - start

        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)

        log.info(
            "http.request",
            method=request.method,
            path=endpoint,
            status=response.status_code,
            duration_ms=f"{duration * 1000:.1f}",
        )
        return response

    # ── Routes ────────────────────────────────────────────────────────────────
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import RedirectResponse

    demo_dir = Path(__file__).resolve().parent.parent.parent / "demo"
    if demo_dir.exists():
        app.mount("/demo", StaticFiles(directory=str(demo_dir), html=True), name="demo")

    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/demo" if demo_dir.exists() else "/docs")

    app.include_router(webhook_router, prefix="/webhooks", tags=["webhooks"])
    app.include_router(api_router, prefix="/api/v1", tags=["api"])

    @app.get("/health", tags=["health"])
    @app.get("/health/live", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "1.0.0"}

    @app.get("/health/ready", tags=["health"])
    async def health_ready() -> JSONResponse:
        from sqlalchemy import text
        from src.db.database import _engine

        components = {}
        all_ok = True

        # Database check
        try:
            if _engine is not None:
                async with _engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                components["database"] = "ok"
            else:
                components["database"] = "ok"
        except Exception:
            components["database"] = "failed"
            all_ok = False

        # Redis & Worker check
        try:
            redis = await get_redis_connection()
            await redis.ping()
            components["redis"] = "ok"

            try:
                workers = await redis.keys("rq:worker:*")
                if not workers:
                    workers = await redis.keys(b"rq:worker:*")
                if workers:
                    components["worker"] = "ok"
                else:
                    components["worker"] = "failed"
                    all_ok = False
            except Exception:
                components["worker"] = "failed"
                all_ok = False
        except Exception:
            components["redis"] = "failed"
            components["worker"] = "failed"
            all_ok = False

        status_str = "ok" if all_ok else "unavailable"
        status_code = 200 if all_ok else 503
        return JSONResponse(
            status_code=status_code,
            content={"status": status_str, "components": components},
        )

    @app.get("/metrics", tags=["observability"])
    async def metrics() -> Response:
        """Prometheus scrape endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


# Instantiate for uvicorn
app = create_app()
