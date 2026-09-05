from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Response
from starlette.requests import Request

from src.api.security import (
    _CSP_POLICY,
    RequestValidationMiddleware,
    SecurityHeadersMiddleware,
)


def _make_request(
    *,
    method: str = "GET",
    path: str = "/api/ping",
    headers: dict[str, str] | None = None,
) -> Request:
    raw_headers = [(name.lower().encode("ascii"), value.encode("ascii")) for name, value in (headers or {}).items()]
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "https",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": raw_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
        "root_path": "",
    }

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


class TestSecurityHeadersMiddleware:
    @pytest.mark.asyncio
    async def test_adds_security_headers_and_strips_server_headers(self) -> None:
        middleware = SecurityHeadersMiddleware(app=AsyncMock(), is_production=True)
        request = _make_request(
            path="/api/ping",
            headers={"X-Request-ID": "req-123"},
        )
        response = Response(
            content="ok",
            headers={"server": "uvicorn", "x-powered-by": "fastapi"},
        )
        call_next = AsyncMock(return_value=response)

        result = await middleware.dispatch(request, call_next)

        assert result.headers["X-Request-ID"] == "req-123"
        assert result.headers["Content-Security-Policy"] == _CSP_POLICY
        assert result.headers["Strict-Transport-Security"] == ("max-age=63072000; includeSubDomains; preload")
        assert result.headers["X-Content-Type-Options"] == "nosniff"
        assert result.headers["X-Frame-Options"] == "DENY"
        assert "server" not in result.headers
        assert "x-powered-by" not in result.headers

    @pytest.mark.asyncio
    async def test_metrics_path_skips_csp(self) -> None:
        middleware = SecurityHeadersMiddleware(app=AsyncMock(), is_production=False)
        request = _make_request(path="/metrics")
        response = Response(content="metrics")

        result = await middleware.dispatch(request, AsyncMock(return_value=response))

        assert "Content-Security-Policy" not in result.headers
        assert "Strict-Transport-Security" not in result.headers

    @pytest.mark.asyncio
    async def test_clears_structlog_context_when_handler_raises(self) -> None:
        middleware = SecurityHeadersMiddleware(app=AsyncMock(), is_production=False)
        request = _make_request(path="/api/ping")

        with (
            patch("src.api.security.uuid.uuid4", return_value="generated-id"),
            patch("src.api.security.structlog.contextvars.bind_contextvars") as bind_mock,
            patch("src.api.security.structlog.contextvars.clear_contextvars") as clear_mock,
            pytest.raises(RuntimeError, match="boom"),
        ):
            await middleware.dispatch(
                request,
                AsyncMock(side_effect=RuntimeError("boom")),
            )

        bind_mock.assert_called_once_with(request_id="generated-id")
        clear_mock.assert_called_once()


class TestRequestValidationMiddleware:
    @pytest.mark.asyncio
    async def test_rejects_invalid_content_length(self) -> None:
        middleware = RequestValidationMiddleware(app=AsyncMock())
        request = _make_request(
            method="POST",
            headers={
                "content-length": "abc",
                "content-type": "application/json",
            },
        )
        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 400
        assert b"Invalid Content-Length header" in response.body
        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejects_oversized_request(self) -> None:
        middleware = RequestValidationMiddleware(app=AsyncMock())
        request = _make_request(
            method="POST",
            headers={
                "content-length": str(RequestValidationMiddleware.MAX_BODY_BYTES + 1),
                "content-type": "application/json",
            },
        )
        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 413
        assert b"Request body too large" in response.body
        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejects_non_json_api_requests(self) -> None:
        middleware = RequestValidationMiddleware(app=AsyncMock())
        request = _make_request(
            method="PATCH",
            path="/api/issues/42",
            headers={
                "content-length": "10",
                "content-type": "text/plain",
            },
        )
        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 415
        assert b"Content-Type must be application/json" in response.body
        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allows_json_requests_with_parameters(self) -> None:
        middleware = RequestValidationMiddleware(app=AsyncMock())
        request = _make_request(
            method="POST",
            path="/api/issues/42",
            headers={
                "content-length": "10",
                "content-type": "application/json; charset=utf-8",
            },
        )
        downstream_response = Response(content="ok")
        call_next = AsyncMock(return_value=downstream_response)

        response = await middleware.dispatch(request, call_next)

        assert response is downstream_response
        call_next.assert_awaited_once()
