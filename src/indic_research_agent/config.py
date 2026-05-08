"""Typed application configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Environment-backed application settings.

    Provider-specific LiteLLM API keys can still be supplied through their
    native environment variables, such as OPENAI_API_KEY or ANTHROPIC_API_KEY.
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

    litellm_model: str = "openai/gpt-4o-mini"
    litellm_custom_llm_provider: str | None = None
    litellm_api_key: SecretStr | None = None
    litellm_api_base: str | None = None
    litellm_temperature: float = 0.2
    litellm_timeout_seconds: float = 30.0
    litellm_max_tokens: int | None = 1200
    litellm_streaming: bool = True

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


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
