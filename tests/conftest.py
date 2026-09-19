"""
tests/conftest.py — Shared pytest fixtures.
"""

import os

import pytest

# Ensure Git never interactively prompts for passwords on stdin during tests
os.environ["GIT_TERMINAL_PROMPT"] = "0"
os.environ["GIT_ASKPASS"] = "echo"

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
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    monkeypatch.setenv("GIT_ASKPASS", "echo")

    # Invalidate settings cache after patching
    from src.config import invalidate_settings_cache

    invalidate_settings_cache()
    yield
    invalidate_settings_cache()


@pytest.fixture(autouse=True)
def prevent_network_git_clones(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure unmocked git.Repo.clone_from in unit tests never makes outbound network calls."""
    import git

    orig_clone_from = git.Repo.clone_from

    def _safe_clone_from(url: str, to_path: str, *args: object, **kwargs: object) -> git.Repo:
        if os.path.exists(url):
            return orig_clone_from(url, to_path, *args, **kwargs)
        raise RuntimeError(f"Hermetic test blocked outbound git clone to: {url}")

    monkeypatch.setattr("git.Repo.clone_from", _safe_clone_from)
