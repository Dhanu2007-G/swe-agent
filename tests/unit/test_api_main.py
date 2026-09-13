from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.testclient import TestClient


def _make_request(path: str = "/boom") -> Request:
    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "https",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 443),
            "root_path": "",
        },
        receive=receive,
    )


class TestCreateApp:
    def test_production_app_adds_trusted_host_and_disables_docs(self) -> None:
        from src.api.main import create_app

        settings = SimpleNamespace(
            is_production=True,
            allowed_hosts=["api.example.com"],
            allowed_cors_origins=["https://api.example.com"],
        )

        with patch("src.api.main.get_settings", return_value=settings):
            app = create_app()

        assert app.docs_url is None
        assert any(middleware.cls is TrustedHostMiddleware for middleware in app.user_middleware)

    async def test_global_exception_handler_returns_internal_server_error(self) -> None:
        from src.api.main import create_app

        settings = SimpleNamespace(
            is_production=False,
            allowed_hosts=["localhost"],
        )

        with (
            patch("src.api.main.get_settings", return_value=settings),
            patch("src.api.main.log.error") as log_error,
        ):
            app = create_app()
            handler = app.exception_handlers[Exception]
            response = await handler(_make_request(), RuntimeError("boom"))

        assert response.status_code == 500
        assert response.body == b'{"detail":"Internal server error"}'
        log_error.assert_called_once()


class TestHealthEndpoints:
    def test_health_live(self) -> None:
        from src.api.main import create_app

        settings = SimpleNamespace(is_production=False, allowed_hosts=["localhost"])
        with patch("src.api.main.get_settings", return_value=settings):
            app = create_app()

        client = TestClient(app)
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "version": "1.0.0"}

    @patch("src.db.database._engine")
    @patch("src.api.main.get_redis_connection")
    def test_health_ready_all_ok(self, mock_get_redis, mock_engine) -> None:
        from src.api.main import create_app

        settings = SimpleNamespace(is_production=False, allowed_hosts=["localhost"])
        with patch("src.api.main.get_settings", return_value=settings):
            app = create_app()

        # mock async context manager for db conn
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_engine.connect.return_value.__aenter__.return_value = mock_conn

        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.keys.return_value = [b"rq:worker:123"]
        mock_get_redis.return_value = mock_redis

        client = TestClient(app)
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json() == {
            "status": "ok",
            "components": {"database": "ok", "redis": "ok", "worker": "ok"},
        }

    @patch("src.db.database._engine")
    @patch("src.api.main.get_redis_connection")
    def test_health_ready_db_fails(self, mock_get_redis, mock_engine) -> None:
        from src.api.main import create_app

        settings = SimpleNamespace(is_production=False, allowed_hosts=["localhost"])
        with patch("src.api.main.get_settings", return_value=settings):
            app = create_app()

        # force DB fail
        mock_engine.connect.side_effect = Exception("DB offline")

        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.keys.return_value = [b"rq:worker:123"]
        mock_get_redis.return_value = mock_redis

        client = TestClient(app)
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "unavailable"
        assert resp.json()["components"]["database"] == "failed"

    @patch("src.db.database._engine")
    @patch("src.api.main.get_redis_connection")
    def test_health_ready_redis_fails_and_worker_empty(self, mock_get_redis, mock_engine) -> None:
        from src.api.main import create_app

        settings = SimpleNamespace(is_production=False, allowed_hosts=["localhost"])
        with patch("src.api.main.get_settings", return_value=settings):
            app = create_app()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_engine.connect.return_value.__aenter__.return_value = mock_conn

        mock_get_redis.side_effect = Exception("Redis offline")

        client = TestClient(app)
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "unavailable"
        assert resp.json()["components"]["redis"] == "failed"
        assert resp.json()["components"]["worker"] == "failed"

    @patch("src.db.database._engine")
    @patch("src.api.main.get_redis_connection")
    def test_health_ready_worker_empty(self, mock_get_redis, mock_engine) -> None:
        from src.api.main import create_app

        settings = SimpleNamespace(is_production=False, allowed_hosts=["localhost"])
        with patch("src.api.main.get_settings", return_value=settings):
            app = create_app()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_engine.connect.return_value.__aenter__.return_value = mock_conn

        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.keys.return_value = []  # no workers
        mock_get_redis.return_value = mock_redis

        client = TestClient(app)
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "unavailable"
        assert resp.json()["components"]["worker"] == "failed"

    def test_metrics_code(self) -> None:
        from src.api.main import create_app

        settings = SimpleNamespace(is_production=False, allowed_hosts=["localhost"])
        with patch("src.api.main.get_settings", return_value=settings):
            app = create_app()

        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    @patch("src.api.main.get_redis_connection")
    @patch("src.api.main.init_db")
    @patch("src.api.main.close_redis_connection")
    async def test_lifespan(self, mock_close_redis, mock_init_db, mock_get_redis) -> None:
        from fastapi import FastAPI

        from src.api.main import lifespan

        app = FastAPI()
        settings = SimpleNamespace(
            is_production=False,
            allowed_hosts=["localhost"],
            app_env="test",
            log_level="INFO",
            log_format="json",
        )
        mock_redis = AsyncMock()
        mock_get_redis.return_value = mock_redis
        with patch("src.api.main.get_settings", return_value=settings):
            async with lifespan(app):
                pass
        mock_redis.ping.assert_awaited_once()
        mock_init_db.assert_awaited_once()
        mock_close_redis.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.api.main.get_redis_connection")
    @patch("src.api.main.init_db")
    async def test_lifespan_tolerates_errors(self, mock_init_db, mock_get_redis) -> None:
        from fastapi import FastAPI

        from src.api.main import lifespan

        app = FastAPI()
        settings = SimpleNamespace(
            is_production=False,
            allowed_hosts=["localhost"],
            app_env="test",
            log_level="INFO",
            log_format="json",
        )
        mock_get_redis.side_effect = RuntimeError("redis fail")
        mock_init_db.side_effect = RuntimeError("db fail")
        with patch("src.api.main.get_settings", return_value=settings):
            async with lifespan(app):
                pass

    @pytest.mark.asyncio
    @patch("src.api.main.get_redis_connection")
    async def test_close_redis_connection(self, mock_get_redis) -> None:
        from src.api.main import close_redis_connection

        mock_redis = AsyncMock()
        mock_get_redis.return_value = mock_redis
        await close_redis_connection()
        mock_redis.aclose.assert_awaited_once()

        # Test close fallback
        class SyncCloseRedis:
            def __init__(self):
                self.closed = False

            async def close(self):
                self.closed = True

        sync_redis = SyncCloseRedis()
        mock_get_redis.return_value = sync_redis
        await close_redis_connection()
        assert sync_redis.closed is True

        # Test exception tolerance
        mock_get_redis.side_effect = RuntimeError("fail")
        await close_redis_connection()

    @pytest.mark.asyncio
    @patch("src.api.main.get_redis_connection")
    @patch("src.api.main.init_db")
    @patch("src.api.main.close_redis_connection", side_effect=RuntimeError("shutdown error"))
    async def test_lifespan_shutdown_error_tolerant(
        self, mock_close_redis, mock_init_db, mock_get_redis
    ) -> None:
        from fastapi import FastAPI

        from src.api.main import lifespan

        app = FastAPI()
        settings = SimpleNamespace(
            is_production=False,
            allowed_hosts=["localhost"],
            app_env="test",
            log_level="INFO",
            log_format="json",
        )
        with patch("src.api.main.get_settings", return_value=settings):
            async with lifespan(app):
                pass

    @patch("src.db.database._engine", None)
    @patch("src.api.main.get_redis_connection")
    def test_health_ready_db_none_and_worker_exception(self, mock_get_redis) -> None:
        from src.api.main import create_app

        settings = SimpleNamespace(is_production=False, allowed_hosts=["localhost"])
        with patch("src.api.main.get_settings", return_value=settings):
            app = create_app()

        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.keys.side_effect = RuntimeError("keys fail")
        mock_get_redis.return_value = mock_redis

        client = TestClient(app)
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["components"]["database"] == "ok"
        assert resp.json()["components"]["worker"] == "failed"

    def test_root_redirect(self) -> None:
        from src.api.main import create_app

        settings = SimpleNamespace(is_production=False, allowed_hosts=["localhost"])
        with patch("src.api.main.get_settings", return_value=settings):
            app = create_app()

        client = TestClient(app, follow_redirects=False)
        resp = client.get("/")
        assert resp.status_code in (307, 302, 200)
