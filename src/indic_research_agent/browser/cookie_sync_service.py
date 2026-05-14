"""Services that keep a disk cookie jar synchronized with CDP."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from indic_research_agent.browser.cdp_client import CdpCookieClient
from indic_research_agent.browser.cookie_jar import CookieJarStore
from indic_research_agent.config import AppSettings, get_settings

logger = logging.getLogger(__name__)


class BrowserCookieClient(Protocol):
    async def __aenter__(self) -> BrowserCookieClient: ...

    async def __aexit__(self, *_exc_info: object) -> None: ...

    async def get_cookies(self) -> list[dict[str, Any]]: ...

    async def set_cookies(self, cookie_params: list[dict[str, Any]]) -> None: ...


BrowserCookieClientFactory = Callable[[str, float], BrowserCookieClient]


class BrowserCookieSyncError(RuntimeError):
    """Raised when cookie sync is configured to fail closed."""


@dataclass(frozen=True)
class BrowserCookieSyncStatus:
    """Sanitized status for logs and CLI JSON output."""

    ok: bool
    action: str
    enabled: bool
    endpoint_configured: bool
    jar_path: str
    cookie_count: int = 0
    message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class BrowserCookieJarService:
    """Apply, snapshot, and watch browser cookies via CDP."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        store: CookieJarStore | None = None,
        client_factory: BrowserCookieClientFactory | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self._settings = settings or get_settings()
        self._store = store or CookieJarStore(self._settings.browser_cookie_jar_path)
        self._client_factory = client_factory or _default_client_factory
        self._sleep = sleep

    @classmethod
    def from_settings(
        cls,
        settings: AppSettings | None = None,
        **kwargs: Any,
    ) -> BrowserCookieJarService:
        return cls(settings or get_settings(), **kwargs)

    async def startup_sync(self) -> BrowserCookieSyncStatus:
        """Ensure the jar, apply jar cookies to CDP, then snapshot active cookies."""

        if not self._settings.browser_cookie_jar_enabled:
            return self._status(ok=True, action="disabled", cookie_count=0)

        self._store.ensure_exists()
        endpoint = self._endpoint()
        if endpoint is None:
            return self._handle_missing_endpoint(action="startup_skipped_no_endpoint")

        try:
            async with self._client(endpoint) as client:
                loaded_cookies = self._store.load()
                if loaded_cookies:
                    await client.set_cookies(loaded_cookies)
                active_cookies = await client.get_cookies()
                saved_cookies = self._store.save_from_cdp_cookies(active_cookies)
        except Exception as exc:
            return self._handle_error(action="startup_sync_failed", exc=exc)

        logger.info(
            "Browser cookie startup sync completed: "
            "applied_count=%d saved_count=%d jar_path=%s",
            len(loaded_cookies),
            len(saved_cookies),
            self._settings.browser_cookie_jar_path,
        )
        return self._status(
            ok=True,
            action="startup_sync",
            cookie_count=len(saved_cookies),
        )

    async def sync_from_browser(
        self,
        client: BrowserCookieClient | None = None,
        *,
        action: str = "sync_from_browser",
    ) -> BrowserCookieSyncStatus:
        """Read active CDP cookies and replace the on-disk jar snapshot."""

        if not self._settings.browser_cookie_jar_enabled:
            return self._status(ok=True, action="disabled", cookie_count=0)

        self._store.ensure_exists()
        if client is not None:
            return await self._sync_from_client(client, action=action)

        endpoint = self._endpoint()
        if endpoint is None:
            return self._handle_missing_endpoint(action=f"{action}_skipped_no_endpoint")

        try:
            async with self._client(endpoint) as managed_client:
                return await self._sync_from_client(managed_client, action=action)
        except Exception as exc:
            return self._handle_error(action=f"{action}_failed", exc=exc)

    @asynccontextmanager
    async def persistent_session(self) -> AsyncIterator[BrowserCookieJarService]:
        """Wrap app-owned browser automation and sync cookies in ``finally``."""

        try:
            yield self
        finally:
            await self.sync_from_browser(action="persistent_session_final_sync")

    async def run_watch_loop(
        self,
        *,
        stop_event: asyncio.Event | None = None,
    ) -> BrowserCookieSyncStatus:
        """Continuously sync cookies and perform a final sync on shutdown."""

        if not self._settings.browser_cookie_jar_enabled:
            return self._status(ok=True, action="disabled", cookie_count=0)

        self._store.ensure_exists()
        endpoint = self._endpoint()
        if endpoint is None:
            return self._handle_missing_endpoint(action="watch_skipped_no_endpoint")

        last_status = self._status(ok=True, action="watch_started", cookie_count=0)
        while stop_event is None or not stop_event.is_set():
            try:
                async with self._client(endpoint) as client:
                    loaded_cookies = self._store.load()
                    if loaded_cookies:
                        await client.set_cookies(loaded_cookies)
                    logger.info(
                        "Browser cookie watch connected: applied_count=%d jar_path=%s",
                        len(loaded_cookies),
                        self._settings.browser_cookie_jar_path,
                    )
                    should_reconnect = False
                    try:
                        while stop_event is None or not stop_event.is_set():
                            last_status = await self._sync_from_client(
                                client,
                                action="watch_sync",
                            )
                            if not last_status.ok:
                                should_reconnect = True
                                break
                            await self._sleep_until_next_tick(stop_event)
                    finally:
                        with suppress(Exception):
                            last_status = await self._sync_from_client(
                                client,
                                action="watch_final_sync",
                            )
                    if stop_event is not None and stop_event.is_set():
                        return last_status
                    if should_reconnect or not last_status.ok:
                        continue
                    return last_status
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_status = self._handle_error(action="watch_sync_failed", exc=exc)
                if stop_event is not None and stop_event.is_set():
                    return last_status
                await self._sleep_until_next_tick(stop_event)
        return last_status

    async def _sync_from_client(
        self,
        client: BrowserCookieClient,
        *,
        action: str,
    ) -> BrowserCookieSyncStatus:
        try:
            active_cookies = await client.get_cookies()
            saved_cookies = self._store.save_from_cdp_cookies(active_cookies)
        except Exception as exc:
            return self._handle_error(action=f"{action}_failed", exc=exc)

        logger.info(
            "Browser cookie sync completed: action=%s saved_count=%d jar_path=%s",
            action,
            len(saved_cookies),
            self._settings.browser_cookie_jar_path,
        )
        return self._status(ok=True, action=action, cookie_count=len(saved_cookies))

    async def _sleep_until_next_tick(
        self,
        stop_event: asyncio.Event | None,
    ) -> None:
        interval = self._settings.browser_cookie_jar_sync_interval_seconds
        if stop_event is None:
            await self._sleep(interval)
            return
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            return

    def _client(self, endpoint: str) -> BrowserCookieClient:
        return self._client_factory(
            endpoint,
            self._settings.browser_cookie_jar_startup_timeout_seconds,
        )

    def _endpoint(self) -> str | None:
        endpoint = self._settings.browser_cdp_endpoint
        if endpoint is None or not endpoint.strip():
            return None
        return endpoint.strip()

    def _handle_missing_endpoint(self, *, action: str) -> BrowserCookieSyncStatus:
        message = "Browser CDP endpoint is not configured"
        if self._settings.browser_cookie_jar_fail_on_error:
            raise BrowserCookieSyncError(message)
        logger.info(
            "Browser cookie sync skipped: action=%s reason=no_cdp_endpoint jar_path=%s",
            action,
            self._settings.browser_cookie_jar_path,
        )
        return self._status(
            ok=True,
            action=action,
            cookie_count=len(self._store.load()),
        )

    def _handle_error(self, *, action: str, exc: Exception) -> BrowserCookieSyncStatus:
        message = (
            f"Browser cookie sync action {action} failed: {exc.__class__.__name__}"
        )
        if self._settings.browser_cookie_jar_fail_on_error:
            raise BrowserCookieSyncError(message) from exc
        logger.warning(message)
        return self._status(ok=False, action=action, cookie_count=0, message=message)

    def _status(
        self,
        *,
        ok: bool,
        action: str,
        cookie_count: int,
        message: str | None = None,
    ) -> BrowserCookieSyncStatus:
        return BrowserCookieSyncStatus(
            ok=ok,
            action=action,
            enabled=self._settings.browser_cookie_jar_enabled,
            endpoint_configured=self._endpoint() is not None,
            jar_path=str(self._settings.browser_cookie_jar_path),
            cookie_count=cookie_count,
            message=message,
        )


def _default_client_factory(
    endpoint: str,
    timeout_seconds: float,
) -> BrowserCookieClient:
    return CdpCookieClient(endpoint, timeout_seconds=timeout_seconds)
