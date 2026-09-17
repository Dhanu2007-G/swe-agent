"""
src/api/security.py — Production security middleware.

Injects:
  - X-Request-ID: UUID per request (for log correlation)
  - Strict-Transport-Security: HSTS in production
  - X-Content-Type-Options: nosniff
  - X-Frame-Options: DENY
  - Content-Security-Policy: restrictive policy
  - Referrer-Policy: strict-origin-when-cross-origin
  - X-XSS-Protection: disabled (CSP supersedes it; old IE guard)
  - Server: stripped (don't leak stack info)

Also validates content-type on POST requests to prevent MIME-type confusion.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog
from fastapi import HTTPException, Request, Response, Security, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.types import ASGIApp

log = structlog.get_logger(__name__)

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-XSS-Protection": "0",  # CSP supersedes; "1" causes issues in modern browsers
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}

_CSP_POLICY = (
    "default-src 'none'; "
    "script-src 'none'; "
    "style-src 'none'; "
    "img-src 'none'; "
    "connect-src 'none'; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
    "form-action 'none'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Injects security headers on every response.
    Strips the Server header to avoid leaking implementation details.
    """

    def __init__(self, app: ASGIApp, is_production: bool = False) -> None:
        super().__init__(app)
        self._is_production = is_production

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Inject request ID for log correlation
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Bind to structlog context for this request
        structlog.contextvars.bind_contextvars(request_id=request_id)

        try:
            response = await call_next(request)

            # Core security headers
            for header, value in _SECURITY_HEADERS.items():
                response.headers[header] = value

            # CSP — only for API responses (not metrics endpoint)
            if not request.url.path.startswith("/metrics"):
                response.headers["Content-Security-Policy"] = _CSP_POLICY

            # HSTS — production only (breaks local dev with HTTPS)
            if self._is_production:
                response.headers["Strict-Transport-Security"] = (
                    "max-age=63072000; includeSubDomains; preload"
                )

            # Strip server identification
            if "server" in response.headers:
                del response.headers["server"]
            if "x-powered-by" in response.headers:
                del response.headers["x-powered-by"]

            # Propagate request ID to client
            response.headers["X-Request-ID"] = request_id

            return response
        finally:
            structlog.contextvars.clear_contextvars()


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Validates incoming requests before they reach route handlers.
    Rejects oversized bodies and incorrect content-types on write endpoints.
    """

    MAX_BODY_BYTES = 1_048_576  # 1 MB — GitHub webhook payloads are typically < 50KB

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Check Content-Length before reading body
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.MAX_BODY_BYTES:
                    log.warning(
                        "security.request_too_large",
                        content_length=content_length,
                        path=request.url.path,
                    )
                    return Response(
                        content='{"detail": "Request body too large"}',
                        status_code=413,
                        media_type="application/json",
                    )
            except ValueError:
                return Response(
                    content='{"detail": "Invalid Content-Length header"}',
                    status_code=400,
                    media_type="application/json",
                )

        # Require Content-Type: application/json on POST/PUT/PATCH
        if request.method in ("POST", "PUT", "PATCH"):
            content_type = request.headers.get("content-type", "")
            # Webhooks from GitHub use application/json but also send extra params
            # So we check prefix only
            if request.url.path.startswith("/api/") and not content_type.startswith(
                "application/json"
            ):
                log.warning(
                    "security.wrong_content_type",
                    content_type=content_type,
                    path=request.url.path,
                )
                return Response(
                    content='{"detail": "Content-Type must be application/json"}',
                    status_code=415,
                    media_type="application/json",
                )

        return await call_next(request)


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
) -> str | None:
    """Validate X-API-Key header against configured API keys."""
    from src.config import get_settings

    settings = get_settings()
    if not getattr(settings, "api_auth_enabled", True):
        return None

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )

    import secrets

    configured_keys = getattr(settings, "api_keys", [])
    valid = any(
        secrets.compare_digest(api_key, configured_key) for configured_key in configured_keys
    )
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return api_key
