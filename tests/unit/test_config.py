from __future__ import annotations

from pydantic import SecretStr

from src.config import Settings


class TestSettings:
    def test_fix_postgres_scheme_handles_legacy_schemes(self) -> None:
        assert Settings.fix_postgres_scheme("postgres://user:pass@db/app") == "postgresql+asyncpg://user:pass@db/app"
        assert Settings.fix_postgres_scheme("postgresql://user:pass@db/app") == "postgresql+asyncpg://user:pass@db/app"

    def test_github_token_value_returns_secret_value(self) -> None:
        settings = Settings.model_construct(
            anthropic_api_key=SecretStr("anthropic"),
            github_token=SecretStr("github-token"),
            github_webhook_secret=SecretStr("webhook-secret"),
            database_url="postgresql+asyncpg://user:pass@db/app",
        )

        assert settings.github_token_value == "github-token"
        assert settings.anthropic_api_key_value == "anthropic"

        settings_multi = Settings.model_construct(
            gemini_api_key=SecretStr("gemini-key"),
            openai_api_key=SecretStr("openai-key"),
        )
        assert settings_multi.gemini_api_key_value == "gemini-key"
        assert settings_multi.openai_api_key_value == "openai-key"

    def test_invalidate_settings_cache(self) -> None:
        from src.config import get_settings, invalidate_settings_cache

        settings_1 = get_settings()
        invalidate_settings_cache()
        settings_2 = get_settings()
        assert settings_1 is not settings_2
