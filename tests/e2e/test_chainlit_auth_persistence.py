from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar

import pytest

pytestmark = pytest.mark.e2e


def test_chainlit_auth_and_persistence_flags() -> None:
    base_url = os.environ.get("CHAINLIT_BASE_URL", "http://localhost:8000").rstrip("/")
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(CookieJar())
    )

    auth_config = _json_get(opener, f"{base_url}/auth/config")
    if auth_config is None:
        pytest.skip(f"Chainlit app is not reachable at {base_url}")

    assert auth_config["requireLogin"] is True
    assert auth_config["passwordAuth"] is True

    login_body = urllib.parse.urlencode(
        {"username": "test", "password": "test1234"}
    ).encode()
    login_request = urllib.request.Request(
        f"{base_url}/login",
        data=login_body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with opener.open(login_request, timeout=10) as response:
        login_payload = json.loads(response.read().decode("utf-8"))
    assert login_payload["success"] is True

    user = _json_get(opener, f"{base_url}/user")
    assert user is not None
    assert user["identifier"] == "test"

    settings = _json_get(opener, f"{base_url}/project/settings?language=en-US")
    assert settings is not None
    assert settings["dataPersistence"] is True
    assert settings["threadResumable"] is True


def _json_get(opener: urllib.request.OpenerDirector, url: str) -> dict | None:
    request = urllib.request.Request(url, method="GET")
    try:
        with opener.open(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError):
        return None
