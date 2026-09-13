"""
src/config.py — Production configuration via Pydantic Settings.
All secrets sourced from environment. No hardcoded values.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, PostgresDsn, RedisDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_secret_key: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_hex(32)))
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    worker_concurrency: int = Field(default=3, ge=1, le=20)

    # ── LLM Providers ────────────────────────────────────────────────────────
    llm_provider: Literal["anthropic", "gemini", "openai"] = "anthropic"

    anthropic_api_key: SecretStr = Field(default_factory=lambda: SecretStr("dummy-anthropic-key"))
    anthropic_model: str = "claude-3-5-sonnet-latest"
    anthropic_max_tokens: int = Field(default=8192, ge=1024, le=32768)
    anthropic_timeout_seconds: int = Field(default=120, ge=10, le=600)
    anthropic_max_retries: int = Field(default=3, ge=0, le=10)

    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-1.5-pro"

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── LangSmith ────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = True
    langchain_endpoint: AnyHttpUrl = AnyHttpUrl("https://api.smith.langchain.com")
    langchain_api_key: SecretStr | None = None
    langchain_project: str = "swe-agent-prod"

    # ── GitHub ───────────────────────────────────────────────────────────────
    github_token: SecretStr
    github_webhook_secret: SecretStr
    github_app_id: int | None = None
    github_app_private_key: SecretStr | None = None
    github_bot_username: str = "swe-agent[bot]"
    github_pr_label: str = "automated-pr"

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: PostgresDsn
    database_pool_size: int = Field(default=10, ge=2, le=50)
    database_max_overflow: int = Field(default=20, ge=0, le=100)
    database_echo: bool = False

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: RedisDsn = RedisDsn("redis://redis:6379/0")
    redis_job_timeout: int = Field(default=1800, ge=60)  # 30 min max per job
    redis_result_ttl: int = Field(default=86400, ge=3600)  # 1 day

    # ── Docker Sandbox ───────────────────────────────────────────────────────
    sandbox_image: str = "swe-agent-sandbox:latest"
    sandbox_memory_limit: str = "512m"
    sandbox_cpu_quota: int = Field(default=50000, ge=10000)  # 50% of one CPU
    sandbox_timeout_seconds: int = Field(default=120, ge=30, le=600)
    sandbox_network_disabled: bool = True
    sandbox_workspace_dir: str = "/workspace"

    # ── Deployment ────────────────────────────────────────────────────────────
    allowed_hosts: list[str] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1"],
        description="Allowed hostnames for TrustedHostMiddleware",
    )

    # ── Agent ────────────────────────────────────────────────────────────────
    agent_max_retries: int = Field(default=3, ge=1, le=10)
    agent_max_files_in_context: int = Field(default=10, ge=1, le=30)
    agent_max_context_tokens: int = Field(default=100_000, ge=10_000)
    agent_max_tokens_per_run: int = Field(default=150_000, ge=10_000)
    agent_max_cost_per_run_usd: float = Field(default=3.00, ge=0.5)
    agent_enable_auto_fork: bool = True
    agent_enable_repro_test: bool = True
    agent_pr_draft_on_failure: bool = True
    sandbox_two_phase_deps: bool = True

    # ── Observability ────────────────────────────────────────────────────────
    otel_exporter_otlp_endpoint: str | None = None
    prometheus_port: int = Field(default=9090, ge=1024, le=65535)

    @field_validator("database_url", mode="before")
    @classmethod
    def fix_postgres_scheme(cls, v: str) -> str:
        """SQLAlchemy async requires postgresql+asyncpg://"""
        if isinstance(v, str) and v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+asyncpg://", 1)
        if isinstance(v, str) and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def anthropic_api_key_value(self) -> str:
        return self.anthropic_api_key.get_secret_value() if self.anthropic_api_key else ""

    @property
    def gemini_api_key_value(self) -> str:
        return self.gemini_api_key.get_secret_value() if self.gemini_api_key else ""

    @property
    def openai_api_key_value(self) -> str:
        return self.openai_api_key.get_secret_value() if self.openai_api_key else ""

    @property
    def github_token_value(self) -> str:
        return self.github_token.get_secret_value()

    @property
    def github_webhook_secret_value(self) -> str:
        return self.github_webhook_secret.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton. Call invalidate_settings_cache() in tests."""
    return Settings()  # type: ignore[call-arg]


def invalidate_settings_cache() -> None:
    get_settings.cache_clear()
