"""Local authentication service for the Chainlit adapter."""

from __future__ import annotations

from dataclasses import dataclass
from hmac import compare_digest

from indic_research_agent.config import AppSettings, get_settings


@dataclass(frozen=True)
class AppPrincipal:
    """Authenticated application principal independent of UI frameworks."""

    identifier: str
    display_name: str
    metadata: dict[str, str]


class AuthService:
    """Verify local/e2e Chainlit password credentials.

    The default `test` / `test1234` account is intentionally for local and e2e
    use only. Production deployments should configure a real identity provider
    or secret-backed credentials.
    """

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()

    def verify_credentials(self, username: str, password: str) -> AppPrincipal | None:
        """Return an app principal when credentials match, else `None`."""

        if not self._settings.chainlit_auth_enabled:
            return None

        expected_username = self._settings.chainlit_auth_username
        expected_password = self._settings.chainlit_auth_password.get_secret_value()
        username_matches = compare_digest(username, expected_username)
        password_matches = compare_digest(password, expected_password)
        if not (username_matches and password_matches):
            return None

        return AppPrincipal(
            identifier=expected_username,
            display_name=expected_username,
            metadata={"role": "tester", "provider": "password"},
        )
