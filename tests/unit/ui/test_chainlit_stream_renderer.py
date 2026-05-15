from __future__ import annotations

import pytest

from indic_research_agent.services.agent_events import (
    AgentCompleted,
    AgentFailed,
    AgentProgress,
    AgentRunStarted,
    AgentToolFinished,
    AgentToolStarted,
)
from indic_research_agent.ui import chainlit_stream_renderer as renderer_module
from indic_research_agent.ui.chainlit_stream_renderer import (
    ChainlitStreamRenderer,
    event_to_step_spec,
)

pytestmark = pytest.mark.unit


class FakeMessage:
    def __init__(self) -> None:
        self.content = ""
        self.is_error = False
        self.tokens: list[str] = []
        self.update_count = 0

    async def stream_token(self, text: str) -> None:
        self.tokens.append(text)

    async def update(self) -> bool:
        self.update_count += 1
        return True


class FakeStep:
    sent_steps: list[FakeStep] = []

    def __init__(
        self,
        *,
        name: str,
        type: str,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.name = name
        self.type = type
        self.tags = tags
        self.metadata = metadata
        self.input = ""
        self.output = ""
        self.is_error = False
        self.send_count = 0
        self.update_count = 0
        self.remove_count = 0

    async def send(self) -> FakeStep:
        self.send_count += 1
        self.sent_steps.append(self)
        return self

    async def update(self) -> bool:
        self.update_count += 1
        return True

    async def remove(self) -> bool:
        self.remove_count += 1
        return True


@pytest.fixture(autouse=True)
def reset_fake_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeStep.sent_steps = []
    monkeypatch.setattr(renderer_module.cl, "Step", FakeStep)


def test_run_started_maps_to_request_step() -> None:
    spec = event_to_step_spec(
        AgentRunStarted(
            question="What is Hindi OCR?",
            session_id="thread-1",
            user_identifier="test",
        )
    )

    assert spec is not None
    assert spec.name == "Request received"
    assert spec.type == "run"
    assert spec.input == "What is Hindi OCR?"
    assert "thread-1" in spec.output
    assert "test" in spec.output


def test_progress_maps_to_safe_step_type() -> None:
    spec = event_to_step_spec(
        AgentProgress(label="LLM call", detail="starting", step_type="llm")
    )

    assert spec is not None
    assert spec.name == "LLM call"
    assert spec.type == "llm"
    assert spec.output == "starting"


@pytest.mark.asyncio
async def test_renderer_reuses_one_status_step_and_removes_it_on_completion() -> None:
    message = FakeMessage()
    renderer = ChainlitStreamRenderer(message)  # type: ignore[arg-type]

    events = [
        AgentRunStarted(question="Find Hindi OCR", session_id="thread-1"),
        AgentProgress(
            label="Request received",
            detail="Preparing research agent run.",
            step_type="run",
        ),
        AgentProgress(label="LLM call", detail="Messages: 1", step_type="llm"),
        AgentToolStarted(
            name="search",
            args_summary="query=Hindi OCR",
            tool_call_id="search-1",
            arguments={"query": "Hindi OCR"},
        ),
        AgentToolFinished(
            name="search",
            latency_ms=12.0,
            result_summary="1 result(s): Hindi OCR source",
            tool_call_id="search-1",
            arguments={"query": "Hindi OCR"},
        ),
        AgentCompleted(
            answer="Final answer", retrieved_context=["ctx"], tool_call_count=1
        ),
    ]

    for event in events:
        await renderer.render(event)

    assert message.content == "Final answer"
    assert message.update_count == 1
    assert len(FakeStep.sent_steps) == 1
    status_step = FakeStep.sent_steps[0]
    assert status_step.send_count == 1
    assert status_step.update_count >= 3
    assert status_step.remove_count == 1
    assert status_step.name != "Completed"
    assert status_step.metadata and status_step.metadata.get("ephemeral_status") is True


@pytest.mark.asyncio
async def test_renderer_removes_status_step_on_failure() -> None:
    message = FakeMessage()
    renderer = ChainlitStreamRenderer(message)  # type: ignore[arg-type]

    await renderer.render(AgentRunStarted(question="Find Hindi OCR"))
    await renderer.render(
        AgentProgress(label="LLM call", detail="Messages: 1", step_type="llm")
    )
    await renderer.render(AgentFailed(message="provider failed"))

    assert message.content == "Request failed: provider failed"
    assert message.is_error is True
    assert message.update_count == 1
    assert len(FakeStep.sent_steps) == 1
    status_step = FakeStep.sent_steps[0]
    assert status_step.remove_count == 1
    assert status_step.name != "Request failed"


def test_status_text_is_single_line_and_bounded() -> None:
    long_text = "Line one\nLine two\t" + ("x" * 120)

    status = renderer_module._one_line_status(long_text, max_chars=32)

    assert "\n" not in status
    assert "\t" not in status
    assert "  " not in status
    assert len(status) <= 32
    assert status.endswith("…")
