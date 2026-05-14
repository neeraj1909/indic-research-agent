from __future__ import annotations

from pathlib import Path

import pytest

from indic_research_agent.config import AppSettings

pytestmark = pytest.mark.unit

BROWSER_ENV_VARS = [
    "BROWSER_COOKIE_JAR_ENABLED",
    "BROWSER_CDP_ENDPOINT",
    "CDP_ENDPOINT",
    "BROWSER_COOKIE_JAR_PATH",
    "BROWSER_COOKIE_JAR_SYNC_INTERVAL_SECONDS",
    "BROWSER_COOKIE_JAR_STARTUP_TIMEOUT_SECONDS",
    "BROWSER_COOKIE_JAR_FAIL_ON_ERROR",
]


def _clear_browser_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in BROWSER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_browser_cookie_settings_have_safe_local_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_browser_env(monkeypatch)

    settings = AppSettings(_env_file=None)

    assert settings.browser_cookie_jar_enabled is True
    assert settings.browser_cdp_endpoint is None
    assert settings.browser_cookie_jar_path == Path(
        "tmp/browser-cookie-jar/cookies.local.json"
    )
    assert settings.browser_cookie_jar_sync_interval_seconds == 30.0
    assert settings.browser_cookie_jar_startup_timeout_seconds == 10.0
    assert settings.browser_cookie_jar_fail_on_error is False


def test_browser_cookie_settings_parse_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_browser_env(monkeypatch)
    monkeypatch.setenv("BROWSER_COOKIE_JAR_ENABLED", "false")
    monkeypatch.setenv("BROWSER_CDP_ENDPOINT", "http://127.0.0.1:9222")
    monkeypatch.setenv("BROWSER_COOKIE_JAR_PATH", "/data/browser-cookies/cookies.json")
    monkeypatch.setenv("BROWSER_COOKIE_JAR_SYNC_INTERVAL_SECONDS", "5.5")
    monkeypatch.setenv("BROWSER_COOKIE_JAR_STARTUP_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("BROWSER_COOKIE_JAR_FAIL_ON_ERROR", "true")

    settings = AppSettings(_env_file=None)

    assert settings.browser_cookie_jar_enabled is False
    assert settings.browser_cdp_endpoint == "http://127.0.0.1:9222"
    assert settings.browser_cookie_jar_path == Path(
        "/data/browser-cookies/cookies.json"
    )
    assert settings.browser_cookie_jar_sync_interval_seconds == 5.5
    assert settings.browser_cookie_jar_startup_timeout_seconds == 2.0
    assert settings.browser_cookie_jar_fail_on_error is True


def test_legacy_cdp_endpoint_alias_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_browser_env(monkeypatch)
    monkeypatch.setenv("CDP_ENDPOINT", "ws://127.0.0.1:9222/devtools/browser/abc")

    settings = AppSettings(_env_file=None)

    assert settings.browser_cdp_endpoint == "ws://127.0.0.1:9222/devtools/browser/abc"
