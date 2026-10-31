from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from indic_research_agent.config import AppSettings

pytestmark = pytest.mark.unit


def test_decisions_are_default_off_and_diagnostics_are_masked(monkeypatch) -> None:
    monkeypatch.delenv("DECISIONS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = AppSettings(_env_file=None)

    assert settings.decisions_mode == "off"
    assert settings.masked_decisions_config == {
        "mode": "off",
        "model": "gpt-6-luna",
        "has_api_key": False,
        "timeout_seconds": 5.0,
        "max_concurrency": 1,
    }


def test_decisions_settings_validate_mode_and_timeout() -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, decisions_mode="unexpected")
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, decisions_timeout_seconds=0)


def test_openai_key_alias_handles_explicit_fields_and_defaults(monkeypatch) -> None:
    monkeypatch.delenv("INDIC_DECISIONS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    empty = AppSettings(_env_file=None, openai_api_key=None, decisions_api_key=None)
    assert empty.openai_api_key is None and empty.decisions_api_key is None
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
    settings = AppSettings(_env_file=None)
    assert settings.openai_api_key == SecretStr("synthetic-key")
    assert settings.decisions_api_key == SecretStr("synthetic-key")
    explicit = AppSettings(
        _env_file=None,
        openai_api_key=SecretStr("generation"),
        decisions_api_key=SecretStr("decision"),
    )
    assert explicit.openai_api_key == SecretStr("generation")
    assert explicit.decisions_api_key == SecretStr("decision")
