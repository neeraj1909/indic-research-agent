"""Framework-neutral streaming events emitted by the agent service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

StepType = Literal["run", "tool", "llm", "retrieval", "undefined"]


@dataclass(frozen=True)
class AgentRunStarted:
    question: str
    session_id: str | None = None
    user_identifier: str | None = None


@dataclass(frozen=True)
class AgentProgress:
    label: str
    detail: str = ""
    step_type: StepType = "run"


@dataclass(frozen=True)
class AgentToolStarted:
    name: str
    args_summary: str
    tool_call_id: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentToolFinished:
    name: str
    latency_ms: float | None = None
    result_summary: str = ""
    tool_call_id: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    result_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentToken:
    text: str


@dataclass(frozen=True)
class AgentCompleted:
    answer: str
    retrieved_context: list[str]
    tool_call_count: int


@dataclass(frozen=True)
class AgentFailed:
    message: str


type AgentStreamEvent = (
    AgentRunStarted
    | AgentProgress
    | AgentToolStarted
    | AgentToolFinished
    | AgentToken
    | AgentCompleted
    | AgentFailed
)


def event_to_dict(event: AgentStreamEvent) -> dict[str, Any]:
    """Serialize an event for logs/tests without depending on UI frameworks."""

    payload = asdict(event)
    payload["event"] = event.__class__.__name__
    return payload
