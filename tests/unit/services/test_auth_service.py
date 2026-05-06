from __future__ import annotations

import pytest
from pydantic import SecretStr

from indic_research_agent.config import AppSettings
from indic_research_agent.services.auth_service import AuthService

pytestmark = pytest.mark.unit


def test_verify_credentials_returns_principal_for_local_e2e_user() -> None:
    service = AuthService(
        AppSettings(
            chainlit_auth_username="test",
            chainlit_auth_password=SecretStr("test1234"),
        )
    )

    principal = service.verify_credentials("test", "test1234")

    assert principal is not None
    assert principal.identifier == "test"
    assert principal.display_name == "test"
    assert principal.metadata == {"role": "tester", "provider": "password"}


@pytest.mark.parametrize(
    ("username", "password"),
    [("test", "wrong"), ("wrong", "test1234"), ("", "")],
)
def test_verify_credentials_rejects_invalid_credentials(
    username: str,
    password: str,
) -> None:
    service = AuthService(
        AppSettings(
            chainlit_auth_username="test",
            chainlit_auth_password=SecretStr("test1234"),
        )
    )

    assert service.verify_credentials(username, password) is None


def test_verify_credentials_rejects_when_disabled() -> None:
    service = AuthService(
        AppSettings(
            chainlit_auth_enabled=False,
            chainlit_auth_username="test",
            chainlit_auth_password=SecretStr("test1234"),
        )
    )

    assert service.verify_credentials("test", "test1234") is None
