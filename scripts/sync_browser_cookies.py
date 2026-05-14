"""Synchronize the persistent browser cookie jar through CDP.

This script never prints cookie names or values. JSON/text output is limited to
sanitized status fields such as action, counts, path, and whether an endpoint is
configured.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import NoReturn

from indic_research_agent.browser.cookie_sync_service import (
    BrowserCookieJarService,
    BrowserCookieSyncError,
    BrowserCookieSyncStatus,
)
from indic_research_agent.config import get_settings

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--startup",
        action="store_true",
        help="Ensure/apply the jar once, then snapshot active browser cookies.",
    )
    action.add_argument(
        "--once",
        action="store_true",
        help="Snapshot active browser cookies to the jar once without applying first.",
    )
    action.add_argument(
        "--watch",
        action="store_true",
        help="Run the background watch loop until interrupted.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print sanitized JSON status.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level diagnostics on stderr.",
    )
    args = parser.parse_args()
    if not args.startup and not args.once and not args.watch:
        args.startup = True
    return args


def configure_logging(*, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )


async def run(args: argparse.Namespace) -> tuple[int, BrowserCookieSyncStatus]:
    settings = get_settings()
    service = BrowserCookieJarService(settings)
    try:
        if args.watch:
            status = await service.run_watch_loop()
        elif args.once:
            status = await service.sync_from_browser(action="once")
        else:
            status = await service.startup_sync()
    except BrowserCookieSyncError as exc:
        status = BrowserCookieSyncStatus(
            ok=False,
            action="failed",
            enabled=settings.browser_cookie_jar_enabled,
            endpoint_configured=bool(
                settings.browser_cdp_endpoint and settings.browser_cdp_endpoint.strip()
            ),
            jar_path=str(settings.browser_cookie_jar_path),
            cookie_count=0,
            message=str(exc),
        )
        return 1, status
    return 0, status


def print_status(status: BrowserCookieSyncStatus, *, as_json: bool) -> None:
    payload = status.to_dict()
    if as_json:
        print(json.dumps(payload, sort_keys=True))
        return

    message = " ".join(f"{key}={value}" for key, value in payload.items())
    print(message)


def main() -> NoReturn:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    exit_code, status = asyncio.run(run(args))
    print_status(status, as_json=args.json)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
