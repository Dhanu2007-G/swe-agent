"""
tests/conftest.py — Shared pytest fixtures.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import patch


# ── Override settings for all tests ──────────────────────────────────────────

@pytest.fixture(autouse=True)
def override_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject safe test values for all settings so no real secrets are needed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token")
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "test-webhook-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")  # use DB 1 for tests
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("SANDBOX_NETWORK_DISABLED", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    # Invalidate settings cache after patching
    from src.config import invalidate_settings_cache
    invalidate_settings_cache()
    yield
    invalidate_settings_cache()



