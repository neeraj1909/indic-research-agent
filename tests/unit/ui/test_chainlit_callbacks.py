from __future__ import annotations

import pytest

from indic_research_agent.config import get_settings
from indic_research_agent.ui.chainlit_app import (
    _history_from_thread,
    _history_from_thread_steps,
    password_auth_callback,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_password_auth_callback_returns_chainlit_user(monkeypatch) -> None:
    monkeypatch.setenv("CHAINLIT_AUTH_USERNAME", "test")
    monkeypatch.setenv("CHAINLIT_AUTH_PASSWORD", "test1234")
    monkeypatch.setenv("CHAINLIT_AUTH_ENABLED", "true")
    get_settings.cache_clear()
    try:
        user = await password_auth_callback("test", "test1234")
    finally:
        get_settings.cache_clear()

    assert user is not None
    assert user.identifier == "test"
    assert user.display_name == "test"
    assert user.metadata == {"role": "tester", "provider": "password"}


@pytest.mark.asyncio
async def test_password_auth_callback_rejects_bad_credentials(monkeypatch) -> None:
    monkeypatch.setenv("CHAINLIT_AUTH_USERNAME", "test")
    monkeypatch.setenv("CHAINLIT_AUTH_PASSWORD", "test1234")
    get_settings.cache_clear()
    try:
        user = await password_auth_callback("test", "wrong")
    finally:
        get_settings.cache_clear()

    assert user is None


def test_history_from_thread_prefers_metadata_history() -> None:
    history = _history_from_thread(
        {
            "metadata": {
                "message_history": [
                    {"role": "user", "content": "from metadata"},
                ]
            },
            "steps": [
                {"type": "user_message", "output": "from steps"},
            ],
        }
    )

    assert history == [{"role": "user", "content": "from metadata"}]


def test_history_from_thread_steps_recovers_user_and_assistant_messages() -> None:
    history = _history_from_thread_steps(
        [
            {"type": "run", "output": "ignore"},
            {"type": "user_message", "output": "Question"},
            {"type": "assistant_message", "output": "Answer"},
        ]
    )

    assert history == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
