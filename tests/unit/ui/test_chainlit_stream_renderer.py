from __future__ import annotations

import pytest

from indic_research_agent.services.agent_events import AgentProgress, AgentRunStarted
from indic_research_agent.ui.chainlit_stream_renderer import event_to_step_spec

pytestmark = pytest.mark.unit


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
