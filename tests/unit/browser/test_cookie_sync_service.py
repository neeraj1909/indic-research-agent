from __future__ import annotations

import asyncio
import logging
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from indic_research_agent.browser.cookie_jar import CookieJarStore
from indic_research_agent.browser.cookie_sync_service import BrowserCookieJarService
from indic_research_agent.config import AppSettings

pytestmark = pytest.mark.unit


class FakeBrowserClient:
    def __init__(
        self,
        get_responses: list[list[dict[str, Any]]] | None = None,
        *,
        get_error: Exception | None = None,
        on_get: Any = None,
    ) -> None:
        self.set_calls: list[list[dict[str, Any]]] = []
        self.get_calls = 0
        self._get_responses = deque(get_responses or [[]])
        self._get_error = get_error
        self._on_get = on_get

    async def __aenter__(self) -> FakeBrowserClient:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        return None

    async def set_cookies(self, cookie_params: list[dict[str, Any]]) -> None:
        self.set_calls.append([dict(cookie) for cookie in cookie_params])

    async def get_cookies(self) -> list[dict[str, Any]]:
        self.get_calls += 1
        if self._on_get is not None:
            self._on_get(self.get_calls)
        if self._get_error is not None:
            raise self._get_error
        if self._get_responses:
            return self._get_responses.popleft()
        return []


def make_settings(
    jar_path: Path,
    *,
    endpoint: str | None = "ws://127.0.0.1:9222/devtools/browser/abc",
    enabled: bool = True,
    fail_on_error: bool = False,
) -> AppSettings:
    return AppSettings(
        _env_file=None,
        browser_cookie_jar_enabled=enabled,
        browser_cdp_endpoint=endpoint,
        browser_cookie_jar_path=jar_path,
        browser_cookie_jar_sync_interval_seconds=0.01,
        browser_cookie_jar_startup_timeout_seconds=2.0,
        browser_cookie_jar_fail_on_error=fail_on_error,
    )


@pytest.mark.asyncio
async def test_startup_sync_creates_missing_jar_without_endpoint(
    tmp_path: Path,
) -> None:
    jar_path = tmp_path / "cookies.json"
    service = BrowserCookieJarService(make_settings(jar_path, endpoint=None))

    status = await service.startup_sync()

    assert status.ok is True
    assert status.action == "startup_skipped_no_endpoint"
    assert status.endpoint_configured is False
    assert status.cookie_count == 0
    assert jar_path.exists()
    assert CookieJarStore(jar_path).load() == []


@pytest.mark.asyncio
async def test_startup_sync_applies_existing_jar_and_saves_active_cookies(
    tmp_path: Path,
) -> None:
    jar_path = tmp_path / "cookies.json"
    store = CookieJarStore(jar_path)
    jar_cookie = {
        "name": "stored",
        "value": "secret-stored-value",
        "domain": "example.test",
        "path": "/",
    }
    store.save_cookie_params([jar_cookie])
    active_cookie = {
        "name": "active",
        "value": "secret-active-value",
        "domain": "example.test",
        "path": "/",
        "session": True,
        "expires": -1,
    }
    fake_client = FakeBrowserClient(get_responses=[[active_cookie]])
    factory_calls: list[tuple[str, float]] = []

    def factory(endpoint: str, timeout_seconds: float) -> FakeBrowserClient:
        factory_calls.append((endpoint, timeout_seconds))
        return fake_client

    service = BrowserCookieJarService(
        make_settings(jar_path),
        store=store,
        client_factory=factory,
    )

    status = await service.startup_sync()

    assert status.ok is True
    assert status.action == "startup_sync"
    assert status.cookie_count == 1
    assert factory_calls == [("ws://127.0.0.1:9222/devtools/browser/abc", 2.0)]
    assert fake_client.set_calls == [[jar_cookie]]
    assert store.load() == [
        {
            "name": "active",
            "value": "secret-active-value",
            "domain": "example.test",
            "path": "/",
        }
    ]


@pytest.mark.asyncio
async def test_sync_from_browser_seeds_and_updates_disk_snapshot(
    tmp_path: Path,
) -> None:
    jar_path = tmp_path / "cookies.json"
    store = CookieJarStore(jar_path)
    active_cookie = {
        "name": "fresh",
        "value": "secret-fresh-value",
        "domain": "example.test",
        "path": "/",
    }
    fake_client = FakeBrowserClient(get_responses=[[active_cookie]])
    service = BrowserCookieJarService(
        make_settings(jar_path),
        store=store,
        client_factory=lambda _endpoint, _timeout: fake_client,
    )

    status = await service.sync_from_browser()

    assert status.ok is True
    assert status.action == "sync_from_browser"
    assert status.cookie_count == 1
    assert store.load() == [active_cookie]


@pytest.mark.asyncio
async def test_persistent_session_syncs_from_browser_when_operation_raises(
    tmp_path: Path,
) -> None:
    jar_path = tmp_path / "cookies.json"
    store = CookieJarStore(jar_path)
    active_cookie = {
        "name": "after-error",
        "value": "secret-after-error-value",
        "domain": "example.test",
        "path": "/",
    }
    fake_client = FakeBrowserClient(get_responses=[[active_cookie]])
    service = BrowserCookieJarService(
        make_settings(jar_path),
        store=store,
        client_factory=lambda _endpoint, _timeout: fake_client,
    )

    with pytest.raises(RuntimeError, match="automation failed"):
        async with service.persistent_session():
            raise RuntimeError("automation failed")

    assert fake_client.get_calls == 1
    assert store.load() == [active_cookie]


@pytest.mark.asyncio
async def test_cookie_sync_errors_do_not_log_cookie_values(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    jar_path = tmp_path / "cookies.json"
    fake_client = FakeBrowserClient(
        get_error=RuntimeError("secret-cookie-value should not be logged")
    )
    service = BrowserCookieJarService(
        make_settings(jar_path),
        client_factory=lambda _endpoint, _timeout: fake_client,
    )
    caplog.set_level(logging.WARNING)

    status = await service.startup_sync()

    assert status.ok is False
    assert "secret-cookie-value" not in (status.message or "")
    assert "secret-cookie-value" not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_watch_loop_performs_final_sync_when_stop_event_is_set(
    tmp_path: Path,
) -> None:
    jar_path = tmp_path / "cookies.json"
    store = CookieJarStore(jar_path)
    stop_event = asyncio.Event()

    def stop_after_first_get(call_count: int) -> None:
        if call_count == 1:
            stop_event.set()

    fake_client = FakeBrowserClient(
        get_responses=[
            [
                {
                    "name": "interval",
                    "value": "secret-interval-value",
                    "domain": "example.test",
                    "path": "/",
                }
            ],
            [
                {
                    "name": "final",
                    "value": "secret-final-value",
                    "domain": "example.test",
                    "path": "/",
                }
            ],
        ],
        on_get=stop_after_first_get,
    )
    service = BrowserCookieJarService(
        make_settings(jar_path),
        store=store,
        client_factory=lambda _endpoint, _timeout: fake_client,
    )

    status = await service.run_watch_loop(stop_event=stop_event)

    assert status.ok is True
    assert status.action == "watch_final_sync"
    assert fake_client.get_calls == 2
    assert store.load() == [
        {
            "name": "final",
            "value": "secret-final-value",
            "domain": "example.test",
            "path": "/",
        }
    ]
