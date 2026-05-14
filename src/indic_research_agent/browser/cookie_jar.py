"""Disk-backed browser cookie jar helpers.

The jar stores CDP ``Network.CookieParam``-compatible dictionaries only. Runtime
``Network.Cookie`` fields such as ``size`` and ``session`` are intentionally
filtered out so the saved snapshot can be passed back to ``Storage.setCookies``.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

COOKIE_JAR_VERSION = 1
COOKIE_JAR_SOURCE = "cdp.Storage.getCookies"
COOKIE_PARAM_FIELDS = frozenset(
    {
        "name",
        "value",
        "url",
        "domain",
        "path",
        "secure",
        "httpOnly",
        "sameSite",
        "expires",
        "priority",
        "sourceScheme",
        "sourcePort",
        "partitionKey",
    }
)
_RUNTIME_COOKIE_FIELDS = frozenset({"size", "session", "partitionKeyOpaque"})


class CookieJarError(ValueError):
    """Raised when a cookie jar cannot be parsed or validated."""


@dataclass(frozen=True)
class CookieJarDocument:
    """Versioned cookie jar document loaded from disk."""

    version: int
    updated_at: str
    source: str
    cookies: list[dict[str, Any]]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "source": self.source,
            "cookies": self.cookies,
        }


class CookieJarStore:
    """Atomic JSON store for browser cookies."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def ensure_exists(self, *, now: datetime | None = None) -> None:
        """Create the parent directory and empty jar file if missing."""

        self._ensure_parent_directory()
        if self.path.exists():
            self._chmod_path(self.path, 0o600)
            return
        self._atomic_write_document(_empty_document(now=now))

    def load(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        """Load non-expired cookie params from the jar."""

        self.ensure_exists(now=now)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CookieJarError(f"Invalid cookie jar JSON at {self.path}") from exc

        document = self._document_from_json(raw)
        return [
            cookie
            for cookie in (
                normalize_cookie_param(item, now=now) for item in document.cookies
            )
            if cookie is not None
        ]

    def save_from_cdp_cookies(
        self,
        cookies: Iterable[Mapping[str, Any]],
        *,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Replace the jar snapshot with cookies read from CDP."""

        cookie_params = [
            cookie_param
            for cookie_param in (
                cdp_cookie_to_cookie_param(cookie, now=now) for cookie in cookies
            )
            if cookie_param is not None
        ]
        self.save_cookie_params(cookie_params, source=COOKIE_JAR_SOURCE, now=now)
        return cookie_params

    def save_cookie_params(
        self,
        cookies: Iterable[Mapping[str, Any]],
        *,
        source: str = COOKIE_JAR_SOURCE,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Replace the jar snapshot with cookie params."""

        cookie_params = [
            cookie_param
            for cookie_param in (
                normalize_cookie_param(cookie, now=now) for cookie in cookies
            )
            if cookie_param is not None
        ]
        self._ensure_parent_directory()
        self._atomic_write_document(
            CookieJarDocument(
                version=COOKIE_JAR_VERSION,
                updated_at=_format_timestamp(now),
                source=source,
                cookies=cookie_params,
            )
        )
        return cookie_params

    def _ensure_parent_directory(self) -> None:
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if parent != Path("."):
            self._chmod_path(parent, 0o700)

    def _atomic_write_document(self, document: CookieJarDocument) -> None:
        self._ensure_parent_directory()
        payload = json.dumps(
            document.to_json_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                temp_path = temp_file.name
                temp_file.write(payload)
                temp_file.write("\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
            self._chmod_path(Path(temp_path), 0o600)
            os.replace(temp_path, self.path)
            self._chmod_path(self.path, 0o600)
        finally:
            if temp_path is not None:
                temp = Path(temp_path)
                if temp.exists():
                    temp.unlink()

    @staticmethod
    def _document_from_json(raw: Any) -> CookieJarDocument:
        if not isinstance(raw, dict):
            raise CookieJarError("Cookie jar root must be a JSON object")
        version = raw.get("version")
        if version != COOKIE_JAR_VERSION:
            raise CookieJarError(f"Unsupported cookie jar version: {version!r}")
        cookies = raw.get("cookies")
        if not isinstance(cookies, list):
            raise CookieJarError("Cookie jar cookies field must be a list")
        updated_at = raw.get("updated_at", "")
        source = raw.get("source", "")
        return CookieJarDocument(
            version=version,
            updated_at=updated_at if isinstance(updated_at, str) else "",
            source=source if isinstance(source, str) else "",
            cookies=cookies,
        )

    @staticmethod
    def _chmod_path(path: Path, mode: int) -> None:
        try:
            path.chmod(mode)
        except OSError:
            # Some filesystems/platforms do not support chmod. The atomic write still
            # leaves a valid jar, and callers/tests can validate modes where supported.
            return


def cdp_cookie_to_cookie_param(
    cookie: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Convert a CDP ``Network.Cookie`` to a ``Network.CookieParam`` dict."""

    return _normalize_cookie(cookie, from_cdp=True, now=now)


def normalize_cookie_param(
    cookie: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Validate/filter an on-disk cookie param dict."""

    return _normalize_cookie(cookie, from_cdp=False, now=now)


def _normalize_cookie(
    cookie: Mapping[str, Any],
    *,
    from_cdp: bool,
    now: datetime | None,
) -> dict[str, Any] | None:
    name = cookie.get("name")
    value = cookie.get("value")
    if not isinstance(name, str) or not isinstance(value, str):
        raise CookieJarError("Cookie entries must include string name and value fields")

    normalized: dict[str, Any] = {"name": name, "value": value}
    for field in COOKIE_PARAM_FIELDS:
        if field in {"name", "value", "expires"}:
            continue
        if field in cookie and cookie[field] is not None:
            normalized[field] = cookie[field]

    expires = cookie.get("expires")
    is_session_cookie = from_cdp and (
        cookie.get("session") is True or _is_session_expires_marker(expires)
    )
    if not is_session_cookie and expires is not None:
        normalized_expires = _normalize_expires(expires)
        if normalized_expires is not None:
            if normalized_expires <= _timestamp(now):
                return None
            normalized["expires"] = normalized_expires

    for runtime_field in _RUNTIME_COOKIE_FIELDS:
        normalized.pop(runtime_field, None)
    return normalized


def _normalize_expires(expires: Any) -> float | None:
    if _is_session_expires_marker(expires):
        return None
    if isinstance(expires, bool):
        raise CookieJarError("Cookie expires field must be numeric, not boolean")
    if isinstance(expires, int | float):
        return float(expires)
    if isinstance(expires, str):
        try:
            return float(expires)
        except ValueError as exc:
            raise CookieJarError("Cookie expires field must be numeric") from exc
    raise CookieJarError("Cookie expires field must be numeric")


def _is_session_expires_marker(expires: Any) -> bool:
    return expires == -1 or expires == "-1"


def _empty_document(*, now: datetime | None = None) -> CookieJarDocument:
    return CookieJarDocument(
        version=COOKIE_JAR_VERSION,
        updated_at=_format_timestamp(now),
        source=COOKIE_JAR_SOURCE,
        cookies=[],
    )


def _format_timestamp(now: datetime | None) -> str:
    value = now or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _timestamp(now: datetime | None) -> float:
    value = now or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.timestamp()
