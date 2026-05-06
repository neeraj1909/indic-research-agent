"""JSON-safe chat history contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

ChatRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatTurn:
    """One persisted, JSON-safe chat turn."""

    role: ChatRole
    content: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


MAX_HISTORY_TURNS = 12
MAX_HISTORY_CHARS = 12_000


def normalize_history(
    raw_history: list[ChatTurn | dict[str, Any]] | None,
) -> list[ChatTurn]:
    """Convert untrusted JSON/session history into valid chat turns."""

    turns: list[ChatTurn] = []
    for item in raw_history or []:
        if isinstance(item, ChatTurn):
            role = item.role
            content = item.content
        elif isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
        else:
            continue
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            turns.append(ChatTurn(role=role, content=content))
    return turns


def history_to_json(history: list[ChatTurn]) -> list[dict[str, str]]:
    """Serialize turns for Chainlit user-session metadata."""

    return [turn.to_dict() for turn in history]


def cap_history(
    history: list[ChatTurn],
    *,
    max_turns: int = MAX_HISTORY_TURNS,
    max_chars: int = MAX_HISTORY_CHARS,
) -> list[ChatTurn]:
    """Keep the most recent turns within a simple context budget."""

    capped: list[ChatTurn] = []
    chars = 0
    for turn in reversed(history[-max_turns:]):
        turn_chars = len(turn.content)
        if capped and chars + turn_chars > max_chars:
            break
        capped.append(turn)
        chars += turn_chars
    return list(reversed(capped))


def append_exchange(
    history: list[ChatTurn],
    *,
    user_content: str,
    assistant_content: str,
) -> list[ChatTurn]:
    """Append a completed user/assistant exchange and cap the result."""

    updated = [
        *history,
        ChatTurn(role="user", content=user_content),
        ChatTurn(role="assistant", content=assistant_content),
    ]
    return cap_history(updated)
