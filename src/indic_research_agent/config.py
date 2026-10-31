"""Typed application configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Environment-backed application settings.

    Direct generation and Decisions use explicitly selected OpenAI credentials.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "indic-research-agent"
    environment: str = "development"
    log_level: str = "INFO"

    openai_model: str = "gpt-4.1-mini"
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENAI_API_KEY"
    )

    openai_temperature: float = 0.2
    openai_timeout_seconds: float = 30.0
    openai_max_tokens: int | None = 1200
    openai_streaming: bool = True

    litellm_model: str = "openai/gpt-4o-mini"
    litellm_custom_llm_provider: str | None = None
    litellm_api_key: SecretStr | None = None
    litellm_api_base: str | None = None
    litellm_temperature: float = 0.2
    litellm_timeout_seconds: float = 30.0
    litellm_max_tokens: int | None = 1200
    litellm_streaming: bool = True

    decisions_mode: Literal["off", "shadow", "active"] = "off"
    decisions_model: str = "gpt-6-luna"
    decisions_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="INDIC_DECISIONS_API_KEY",
    )
    # The local live evidence observed a successful response in ~1.97s. Keep
    # enough headroom for normal transport overhead while remaining bounded.
    decisions_timeout_seconds: float = Field(default=5.0, gt=0, le=10)
    decisions_max_concurrency: int = Field(default=1, gt=0, le=8)

    query_kit_providers: str = Field(
        default="semantic-scholar,semantic-scholar-web,pubmed,arxiv-web"
    )
    query_kit_timeout_seconds: float = 30.0

    phoenix_enabled: bool = False
    phoenix_collector_endpoint: str = "http://10.20.30.1:16006"
    phoenix_project_name: str = "indic-research-agent"
    phoenix_protocol: str = "http/protobuf"
    phoenix_batch_spans: bool = True
    phoenix_auto_instrument: bool = True

    database_url: str = Field(
        default=(
            "postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent"
        ),
        validation_alias=AliasChoices("APP_DATABASE_URL", "DATABASE_URL"),
    )
    redis_url: str = "redis://localhost:6379/0"

    chainlit_auth_enabled: bool = True
    chainlit_auth_secret: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("CHAINLIT_AUTH_SECRET"),
    )
    chainlit_auth_username: str = "test"
    chainlit_auth_password: SecretStr = SecretStr("test1234")
    chainlit_database_url: str = Field(
        default=(
            "postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent"
        ),
        validation_alias=AliasChoices("CHAINLIT_DATABASE_URL"),
    )
    chainlit_database_schema: str = "chainlit"
    chainlit_data_layer_show_logger: bool = False

    browser_cookie_jar_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("BROWSER_COOKIE_JAR_ENABLED"),
    )
    browser_cdp_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("BROWSER_CDP_ENDPOINT", "CDP_ENDPOINT"),
    )
    browser_cookie_jar_path: Path = Field(
        default=Path("tmp/browser-cookie-jar/cookies.local.json"),
        validation_alias=AliasChoices("BROWSER_COOKIE_JAR_PATH"),
    )
    browser_cookie_jar_sync_interval_seconds: float = Field(
        default=30.0,
        gt=0,
        validation_alias=AliasChoices("BROWSER_COOKIE_JAR_SYNC_INTERVAL_SECONDS"),
    )
    browser_cookie_jar_startup_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        validation_alias=AliasChoices("BROWSER_COOKIE_JAR_STARTUP_TIMEOUT_SECONDS"),
    )
    browser_cookie_jar_fail_on_error: bool = Field(
        default=False,
        validation_alias=AliasChoices("BROWSER_COOKIE_JAR_FAIL_ON_ERROR"),
    )

    @model_validator(mode="after")
    def _default_decisions_key(self) -> AppSettings:
        if self.decisions_api_key is None:
            self.decisions_api_key = self.openai_api_key
        return self

    @property
    def query_kit_provider_ids(self) -> list[str]:
        return [
            provider.strip()
            for provider in self.query_kit_providers.split(",")
            if provider.strip()
        ] or ["all"]

    @property
    def masked_litellm_config(self) -> dict[str, object]:
        return {
            "model": self.litellm_model,
            "custom_llm_provider": self.litellm_custom_llm_provider,
            "has_api_base": self.litellm_api_base is not None,
            "has_api_key": self.litellm_api_key is not None,
            "temperature": self.litellm_temperature,
            "timeout_seconds": self.litellm_timeout_seconds,
            "max_tokens": self.litellm_max_tokens,
            "streaming": self.litellm_streaming,
        }

    @property
    def masked_decisions_config(self) -> dict[str, object]:
        return {
            "mode": self.decisions_mode,
            "model": self.decisions_model,
            "has_api_key": self.decisions_api_key is not None,
            "timeout_seconds": self.decisions_timeout_seconds,
            "max_concurrency": self.decisions_max_concurrency,
        }


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
