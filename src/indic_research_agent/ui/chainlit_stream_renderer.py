"""Render framework-neutral agent events in Chainlit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import chainlit as cl

from indic_research_agent.services.agent_events import (
    AgentCompleted,
    AgentFailed,
    AgentProgress,
    AgentRunStarted,
    AgentStreamEvent,
    AgentToken,
    AgentToolFinished,
    AgentToolStarted,
)

ChainlitStepType = Literal["run", "tool", "llm", "retrieval", "undefined"]
_STEP_TYPES = {"run", "tool", "llm", "retrieval", "undefined"}


@dataclass(frozen=True)
class StepSpec:
    name: str
    type: ChainlitStepType
    input: str = ""
    output: str = ""
    is_error: bool = False
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class ChainlitStreamRenderer:
    """Small Chainlit presentation adapter for agent stream events."""

    def __init__(self, message: cl.Message) -> None:
        self._message = message
        self._status_step: cl.Step | None = None
        self._token_seen = False
        self.completed_answer: str | None = None
        self.failed = False

    async def render(self, event: AgentStreamEvent) -> None:
        if isinstance(event, AgentToken):
            self._token_seen = True
            await self._message.stream_token(event.text)
            return
        if isinstance(event, AgentCompleted):
            self.completed_answer = event.answer
            if not self._token_seen:
                self._message.content = event.answer
            await self._message.update()
            await self._clear_status()
            return
        if isinstance(event, AgentFailed):
            self.failed = True
            self._message.content = f"Request failed: {event.message}"
            self._message.is_error = True
            await self._message.update()
            await self._clear_status()
            return
        if isinstance(event, AgentToolStarted):
            await self._start_tool_step(event)
            return
        if isinstance(event, AgentToolFinished):
            await self._finish_tool_step(event)
            return
        spec = event_to_step_spec(event)
        if spec is not None:
            await self._update_status(spec)

    async def _start_tool_step(self, event: AgentToolStarted) -> None:
        await self._update_status(
            StepSpec(
                name=f"Tool: {event.name}",
                type=_tool_step_type(event.name),
                input=event.args_summary,
                output="Running…",
                tags=["agent", "tool", event.name],
                metadata={"tool_call_id": event.tool_call_id},
            )
        )

    async def _finish_tool_step(self, event: AgentToolFinished) -> None:
        output = event.result_summary
        if event.latency_ms is not None:
            output = f"{output}; latency: {event.latency_ms:.0f} ms"
        await self._update_status(
            StepSpec(
                name=f"Tool: {event.name}",
                type=_tool_step_type(event.name),
                input=str(event.arguments),
                output=output,
                tags=["agent", "tool", event.name],
                metadata={"tool_call_id": event.tool_call_id},
            )
        )

    async def _update_status(self, spec: StepSpec) -> None:
        metadata = {**(spec.metadata or {}), "ephemeral_status": True}
        tags = _ephemeral_tags(spec.tags)
        name = _one_line_status(spec.name, max_chars=80) or "Working"
        input_value = _one_line_status(spec.input)
        output = _one_line_status(spec.output)

        if self._status_step is None:
            self._status_step = cl.Step(
                name=name,
                type=spec.type,
                tags=tags,
                metadata=metadata,
            )
            self._status_step.input = input_value
            self._status_step.output = output
            self._status_step.is_error = spec.is_error
            await self._status_step.send()
            return

        self._status_step.name = name
        self._status_step.type = spec.type
        self._status_step.tags = tags
        self._status_step.metadata = metadata
        self._status_step.input = input_value
        self._status_step.output = output
        self._status_step.is_error = spec.is_error
        await self._status_step.update()

    async def _clear_status(self) -> None:
        if self._status_step is None:
            return
        status_step = self._status_step
        self._status_step = None
        await status_step.remove()


def event_to_step_spec(event: AgentStreamEvent) -> StepSpec | None:
    """Map service events to testable Chainlit step descriptions."""

    if isinstance(event, AgentRunStarted):
        detail = "Agent run started."
        if event.session_id:
            detail = f"{detail}\nSession: {event.session_id}"
        if event.user_identifier:
            detail = f"{detail}\nUser: {event.user_identifier}"
        return StepSpec(
            name="Request received",
            type="run",
            input=event.question,
            output=detail,
            tags=["agent", "start"],
        )
    if isinstance(event, AgentProgress):
        return StepSpec(
            name=event.label,
            type=_safe_step_type(event.step_type),
            output=event.detail,
            tags=["agent", "progress"],
        )
    return None


def _tool_step_type(tool_name: str) -> ChainlitStepType:
    return "retrieval" if tool_name == "search" else "tool"


def _ephemeral_tags(tags: list[str] | None) -> list[str]:
    values = list(tags or [])
    for tag in ("agent", "progress", "ephemeral"):
        if tag not in values:
            values.append(tag)
    return values


def _one_line_status(text: str, *, max_chars: int = 140) -> str:
    if max_chars <= 0:
        return ""
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    if max_chars == 1:
        return "…"
    return f"{normalized[: max_chars - 1].rstrip()}…"


def _safe_step_type(step_type: str) -> ChainlitStepType:
    if step_type in _STEP_TYPES:
        return step_type  # type: ignore[return-value]
    return "run"
