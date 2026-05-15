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
        self._tool_steps: dict[str, cl.Step] = {}
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
            await self._send_step(
                StepSpec(
                    name="Completed",
                    type="run",
                    output=(
                        f"Retrieved contexts: {len(event.retrieved_context)}; "
                        f"tool calls: {event.tool_call_count}"
                    ),
                    tags=["agent", "complete"],
                )
            )
            return
        if isinstance(event, AgentFailed):
            self.failed = True
            self._message.content = f"Request failed: {event.message}"
            self._message.is_error = True
            await self._message.update()
            await self._send_step(
                StepSpec(
                    name="Request failed",
                    type="run",
                    output=event.message,
                    is_error=True,
                    tags=["agent", "error"],
                )
            )
            return
        if isinstance(event, AgentToolStarted):
            await self._start_tool_step(event)
            return
        if isinstance(event, AgentToolFinished):
            await self._finish_tool_step(event)
            return
        spec = event_to_step_spec(event)
        if spec is not None:
            await self._send_step(spec)

    async def _start_tool_step(self, event: AgentToolStarted) -> None:
        key = event.tool_call_id or event.name
        step = cl.Step(
            name=f"Tool: {event.name}",
            type=_tool_step_type(event.name),
            tags=["agent", "tool", event.name],
            metadata={"tool_call_id": event.tool_call_id},
        )
        step.input = event.args_summary
        step.output = "Running…"
        await step.send()
        self._tool_steps[key] = step

    async def _finish_tool_step(self, event: AgentToolFinished) -> None:
        key = event.tool_call_id or event.name
        step = self._tool_steps.get(key)
        output = event.result_summary
        if event.latency_ms is not None:
            output = f"{output}\n\nLatency: {event.latency_ms:.0f} ms"
        if step is None:
            await self._send_step(
                StepSpec(
                    name=f"Tool: {event.name}",
                    type=_tool_step_type(event.name),
                    input=str(event.arguments),
                    output=output,
                    tags=["agent", "tool", event.name],
                    metadata={"tool_call_id": event.tool_call_id},
                )
            )
            return
        step.output = output
        await step.update()

    async def _send_step(self, spec: StepSpec) -> None:
        step = cl.Step(
            name=spec.name,
            type=spec.type,
            tags=spec.tags,
            metadata=spec.metadata,
        )
        step.input = spec.input
        step.output = spec.output
        step.is_error = spec.is_error
        await step.send()


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


def _safe_step_type(step_type: str) -> ChainlitStepType:
    if step_type in _STEP_TYPES:
        return step_type  # type: ignore[return-value]
    return "run"
