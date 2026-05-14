from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from indic_research_agent.browser.cookie_jar import (
    COOKIE_JAR_SOURCE,
    COOKIE_JAR_VERSION,
    CookieJarStore,
)

pytestmark = pytest.mark.unit

NOW = datetime.fromtimestamp(1_700_000_000, tz=UTC)
FUTURE = 1_800_000_000.0
PAST = 1_600_000_000.0


def test_ensure_exists_creates_empty_versioned_jar_with_restrictive_modes(
    tmp_path: Path,
) -> None:
    jar_path = tmp_path / "browser-cookies" / "cookies.json"
    store = CookieJarStore(jar_path)

    store.ensure_exists(now=NOW)

    assert jar_path.exists()
    payload = json.loads(jar_path.read_text(encoding="utf-8"))
    assert payload == {
        "version": COOKIE_JAR_VERSION,
        "updated_at": "2023-11-14T22:13:20Z",
        "source": COOKIE_JAR_SOURCE,
        "cookies": [],
    }
    if os.name == "posix":
        assert stat.S_IMODE(jar_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(jar_path.stat().st_mode) == 0o600


def test_save_from_cdp_cookies_filters_runtime_fields_and_preserves_sessions(
    tmp_path: Path,
) -> None:
    jar_path = tmp_path / "cookies.json"
    store = CookieJarStore(jar_path)

    saved = store.save_from_cdp_cookies(
        [
            {
                "name": "session",
                "value": "secret-session-value",
                "domain": "example.test",
                "path": "/",
                "secure": True,
                "httpOnly": True,
                "sameSite": "Lax",
                "expires": -1,
                "session": True,
                "size": 123,
                "partitionKeyOpaque": True,
            },
            {
                "name": "persistent",
                "value": "secret-persistent-value",
                "domain": "example.test",
                "path": "/",
                "expires": FUTURE,
                "priority": "Medium",
                "sourceScheme": "Secure",
                "sourcePort": 443,
            },
            {
                "name": "expired",
                "value": "expired-value",
                "domain": "example.test",
                "path": "/",
                "expires": PAST,
            },
        ],
        now=NOW,
    )

    assert saved == [
        {
            "name": "session",
            "value": "secret-session-value",
            "domain": "example.test",
            "path": "/",
            "secure": True,
            "httpOnly": True,
            "sameSite": "Lax",
        },
        {
            "name": "persistent",
            "value": "secret-persistent-value",
            "domain": "example.test",
            "path": "/",
            "expires": FUTURE,
            "priority": "Medium",
            "sourceScheme": "Secure",
            "sourcePort": 443,
        },
    ]
    for cookie in saved:
        assert "size" not in cookie
        assert "session" not in cookie
        assert "partitionKeyOpaque" not in cookie

    assert store.load(now=NOW) == saved


def test_save_is_deterministic_atomic_json_and_replaces_deleted_cookies(
    tmp_path: Path,
) -> None:
    jar_path = tmp_path / "cookies.json"
    store = CookieJarStore(jar_path)

    store.save_from_cdp_cookies(
        [
            {"name": "one", "value": "1", "domain": "example.test", "path": "/"},
            {"name": "two", "value": "2", "domain": "example.test", "path": "/"},
        ],
        now=NOW,
    )
    saved = store.save_from_cdp_cookies(
        [{"name": "two", "value": "2", "domain": "example.test", "path": "/"}],
        now=NOW,
    )

    assert saved == [
        {"name": "two", "value": "2", "domain": "example.test", "path": "/"}
    ]
    assert store.load(now=NOW) == saved
    raw = jar_path.read_text(encoding="utf-8")
    assert (
        raw
        == json.dumps(
            json.loads(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        + "\n"
    )
    assert not list(jar_path.parent.glob("*.tmp"))


def test_load_drops_expired_cookie_params_from_existing_jar(tmp_path: Path) -> None:
    jar_path = tmp_path / "cookies.json"
    jar_path.write_text(
        json.dumps(
            {
                "version": COOKIE_JAR_VERSION,
                "updated_at": "2023-11-14T22:13:20Z",
                "source": COOKIE_JAR_SOURCE,
                "cookies": [
                    {
                        "name": "valid",
                        "value": "secret-valid-value",
                        "domain": "example.test",
                        "path": "/",
                        "expires": FUTURE,
                    },
                    {
                        "name": "expired",
                        "value": "expired-value",
                        "domain": "example.test",
                        "path": "/",
                        "expires": PAST,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert CookieJarStore(jar_path).load(now=NOW) == [
        {
            "name": "valid",
            "value": "secret-valid-value",
            "domain": "example.test",
            "path": "/",
            "expires": FUTURE,
        }
    ]
