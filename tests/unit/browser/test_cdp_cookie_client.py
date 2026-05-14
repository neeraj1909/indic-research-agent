from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from typing import Any

import pytest

from indic_research_agent.browser.cdp_client import (
    CdpCookieClient,
    CdpError,
    discover_websocket_endpoint,
)

pytestmark = pytest.mark.unit


class FakeTransport:
    def __init__(self, responses: list[Mapping[str, Any]]) -> None:
        self.sent: list[dict[str, Any]] = []
        self._responses = deque(json.dumps(response) for response in responses)

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        return self._responses.popleft()


@pytest.mark.asyncio
async def test_discover_websocket_endpoint_returns_direct_ws_url() -> None:
    endpoint = "ws://127.0.0.1:9222/devtools/browser/abc"

    assert await discover_websocket_endpoint(endpoint) == endpoint


@pytest.mark.asyncio
async def test_discover_websocket_endpoint_reads_http_json_version() -> None:
    calls: list[tuple[str, float]] = []

    def fetch_json(url: str, timeout_seconds: float) -> Mapping[str, Any]:
        calls.append((url, timeout_seconds))
        return {"webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/abc"}

    endpoint = await discover_websocket_endpoint(
        "http://127.0.0.1:9222/",
        timeout_seconds=2.5,
        fetch_json=fetch_json,
    )

    assert endpoint == "ws://127.0.0.1:9222/devtools/browser/abc"
    assert calls == [("http://127.0.0.1:9222/json/version", 2.5)]


@pytest.mark.asyncio
async def test_get_and_set_cookies_use_storage_cdp_methods() -> None:
    transport = FakeTransport(
        [
            {
                "id": 1,
                "result": {
                    "cookies": [
                        {"name": "session", "value": "secret", "domain": "example.test"}
                    ]
                },
            },
            {"id": 2, "result": {}},
        ]
    )
    client = CdpCookieClient(
        "ws://127.0.0.1:9222/devtools/browser/abc",
        transport=transport,
        timeout_seconds=1,
    )

    cookies = await client.get_cookies()
    await client.set_cookies(
        [{"name": "session", "value": "secret", "domain": "example.test"}]
    )

    assert cookies == [{"name": "session", "value": "secret", "domain": "example.test"}]
    assert transport.sent == [
        {"id": 1, "method": "Storage.getCookies"},
        {
            "id": 2,
            "method": "Storage.setCookies",
            "params": {
                "cookies": [
                    {"name": "session", "value": "secret", "domain": "example.test"}
                ]
            },
        },
    ]


@pytest.mark.asyncio
async def test_call_ignores_events_until_matching_response_id() -> None:
    transport = FakeTransport(
        [
            {"method": "Storage.cookieChanged", "params": {"reason": "Overwrite"}},
            {"id": 1, "result": {"cookies": []}},
        ]
    )
    client = CdpCookieClient(
        "ws://127.0.0.1:9222/devtools/browser/abc",
        transport=transport,
        timeout_seconds=1,
    )

    assert await client.get_cookies() == []


@pytest.mark.asyncio
async def test_cdp_error_does_not_include_cookie_values() -> None:
    transport = FakeTransport(
        [
            {
                "id": 1,
                "error": {
                    "code": -32000,
                    "message": "Rejected cookie value secret-cookie-value",
                },
            }
        ]
    )
    client = CdpCookieClient(
        "ws://127.0.0.1:9222/devtools/browser/abc",
        transport=transport,
        timeout_seconds=1,
    )

    with pytest.raises(CdpError) as exc_info:
        await client.set_cookies(
            [
                {
                    "name": "session",
                    "value": "secret-cookie-value",
                    "domain": "example.test",
                }
            ]
        )

    message = str(exc_info.value)
    assert "Storage.setCookies" in message
    assert "secret-cookie-value" not in message
    assert "Rejected cookie" not in message


@pytest.mark.asyncio
async def test_unsupported_endpoint_scheme_raises_sanitized_error() -> None:
    with pytest.raises(CdpError, match="Unsupported CDP endpoint scheme"):
        await discover_websocket_endpoint("ftp://127.0.0.1:9222")
