"""Minimal async Chrome DevTools Protocol cookie client."""

from __future__ import annotations

import asyncio
import json
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol
from urllib.parse import urlparse

from websockets.asyncio.client import connect as websocket_connect

JsonFetcher = Callable[[str, float], Mapping[str, Any]]


class CdpTransport(Protocol):
    """Subset of a websocket transport used by the CDP JSON-RPC client."""

    async def send(self, message: str) -> None: ...

    async def recv(self) -> str: ...


class CdpError(RuntimeError):
    """Sanitized CDP client error.

    Cookie payloads and CDP error bodies can contain secrets. Error strings from
    this class intentionally include only method names/status codes, never the
    request params, response body, or cookie values.
    """


class CdpCookieClient:
    """Async JSON-RPC client for CDP Storage cookie methods."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = 10.0,
        transport: CdpTransport | None = None,
        fetch_json: JsonFetcher | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._transport = transport
        self._external_transport = transport is not None
        self._fetch_json = fetch_json
        self._connection_manager: Any | None = None
        self._next_id = 0

    async def __aenter__(self) -> CdpCookieClient:
        await self.connect()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    async def connect(self) -> CdpCookieClient:
        """Open a websocket connection unless a test transport was injected."""

        if self._transport is not None:
            return self
        websocket_endpoint = await discover_websocket_endpoint(
            self._endpoint,
            timeout_seconds=self._timeout_seconds,
            fetch_json=self._fetch_json,
        )
        self._connection_manager = websocket_connect(
            websocket_endpoint,
            open_timeout=self._timeout_seconds,
            close_timeout=min(self._timeout_seconds, 5.0),
        )
        self._transport = await asyncio.wait_for(
            self._connection_manager.__aenter__(),
            timeout=self._timeout_seconds,
        )
        return self

    async def close(self) -> None:
        """Close an owned websocket connection."""

        if self._connection_manager is not None:
            await self._connection_manager.__aexit__(None, None, None)
            self._connection_manager = None
        if not self._external_transport:
            self._transport = None

    async def get_cookies(self) -> list[dict[str, Any]]:
        """Read all browser cookies using ``Storage.getCookies``."""

        result = await self.call("Storage.getCookies")
        cookies = result.get("cookies", [])
        if not isinstance(cookies, list):
            raise CdpError("CDP call Storage.getCookies returned invalid cookies")
        return [dict(cookie) for cookie in cookies if isinstance(cookie, Mapping)]

    async def set_cookies(self, cookie_params: Iterable[Mapping[str, Any]]) -> None:
        """Apply cookie params using ``Storage.setCookies``."""

        await self.call(
            "Storage.setCookies",
            {"cookies": [dict(cookie) for cookie in cookie_params]},
        )

    async def call(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and return its result object."""

        if self._transport is None:
            await self.connect()
        if self._transport is None:
            raise CdpError(f"CDP call {method} failed before connection")

        self._next_id += 1
        request_id = self._next_id
        payload: dict[str, Any] = {"id": request_id, "method": method}
        if params is not None:
            payload["params"] = dict(params)

        try:
            await asyncio.wait_for(
                self._transport.send(
                    json.dumps(payload, sort_keys=True, separators=(",", ":"))
                ),
                timeout=self._timeout_seconds,
            )
            while True:
                raw_response = await asyncio.wait_for(
                    self._transport.recv(),
                    timeout=self._timeout_seconds,
                )
                response = _parse_response(raw_response, method=method)
                if response.get("id") != request_id:
                    continue
                if "error" in response:
                    error = response["error"]
                    code = error.get("code") if isinstance(error, Mapping) else None
                    raise CdpError(f"CDP call {method} failed with code {code!r}")
                result = response.get("result", {})
                if not isinstance(result, dict):
                    raise CdpError(f"CDP call {method} returned invalid result")
                return result
        except CdpError:
            raise
        except TimeoutError as exc:
            raise CdpError(f"CDP call {method} timed out") from exc
        except json.JSONDecodeError as exc:
            raise CdpError(f"CDP call {method} returned invalid JSON") from exc
        except Exception as exc:
            raise CdpError(
                f"CDP call {method} failed: {exc.__class__.__name__}"
            ) from exc


async def discover_websocket_endpoint(
    endpoint: str,
    *,
    timeout_seconds: float = 10.0,
    fetch_json: JsonFetcher | None = None,
) -> str:
    """Resolve a CDP websocket URL from ws/wss or HTTP /json/version endpoint."""

    endpoint = endpoint.strip()
    parsed = urlparse(endpoint)
    if parsed.scheme in {"ws", "wss"}:
        return endpoint
    if parsed.scheme not in {"http", "https"}:
        raise CdpError("Unsupported CDP endpoint scheme")

    version_url = f"{endpoint.rstrip('/')}/json/version"
    fetcher = fetch_json or _fetch_json_from_url
    try:
        payload = await asyncio.wait_for(
            asyncio.to_thread(fetcher, version_url, timeout_seconds),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        raise CdpError("CDP websocket discovery timed out") from exc
    except OSError as exc:
        raise CdpError("CDP websocket discovery failed") from exc

    websocket_url = payload.get("webSocketDebuggerUrl")
    if not isinstance(websocket_url, str) or not websocket_url:
        raise CdpError("CDP websocket discovery returned no browser websocket URL")
    return websocket_url


def _fetch_json_from_url(url: str, timeout_seconds: float) -> Mapping[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise CdpError("CDP websocket discovery returned invalid JSON")
    return payload


def _parse_response(raw_response: str, *, method: str) -> dict[str, Any]:
    payload = json.loads(raw_response)
    if not isinstance(payload, dict):
        raise CdpError(f"CDP call {method} returned invalid response")
    return payload
