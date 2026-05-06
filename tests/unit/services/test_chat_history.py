from __future__ import annotations

import pytest

from indic_research_agent.services.chat_history import (
    ChatTurn,
    append_exchange,
    cap_history,
    history_to_json,
    normalize_history,
)

pytestmark = pytest.mark.unit


def test_normalize_history_filters_invalid_json_values() -> None:
    history = normalize_history(
        [
            {"role": "user", "content": " hello "},
            {"role": "system", "content": "ignore"},
            {"role": "assistant", "content": ""},
            ChatTurn(role="assistant", content="answer"),
        ]
    )

    assert history == [
        ChatTurn(role="user", content="hello"),
        ChatTurn(role="assistant", content="answer"),
    ]


def test_history_to_json_is_json_safe() -> None:
    assert history_to_json([ChatTurn(role="user", content="hello")]) == [
        {"role": "user", "content": "hello"}
    ]


def test_cap_history_keeps_recent_turns_under_budget() -> None:
    history = [
        ChatTurn(role="user", content="older"),
        ChatTurn(role="assistant", content="old answer"),
        ChatTurn(role="user", content="recent question"),
        ChatTurn(role="assistant", content="recent answer"),
    ]

    assert cap_history(history, max_turns=3, max_chars=100) == history[-3:]
    assert cap_history(history, max_turns=4, max_chars=20) == history[-1:]


def test_append_exchange_adds_user_and_assistant_turns() -> None:
    history = append_exchange(
        [ChatTurn(role="user", content="first")],
        user_content="follow up",
        assistant_content="second answer",
    )

    assert history[-2:] == [
        ChatTurn(role="user", content="follow up"),
        ChatTurn(role="assistant", content="second answer"),
    ]
