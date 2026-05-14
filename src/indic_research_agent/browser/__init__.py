"""Browser automation support services."""

from indic_research_agent.browser.cdp_client import CdpCookieClient, CdpError
from indic_research_agent.browser.cookie_jar import CookieJarStore
from indic_research_agent.browser.cookie_sync_service import (
    BrowserCookieJarService,
    BrowserCookieSyncError,
    BrowserCookieSyncStatus,
)

__all__ = [
    "BrowserCookieJarService",
    "BrowserCookieSyncError",
    "BrowserCookieSyncStatus",
    "CdpCookieClient",
    "CdpError",
    "CookieJarStore",
]
