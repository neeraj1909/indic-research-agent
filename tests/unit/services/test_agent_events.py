from __future__ import annotations

import pytest

from indic_research_agent.services.agent_events import (
    AgentCompleted,
    AgentProgress,
    AgentToolFinished,
    AgentToolStarted,
    event_to_dict,
)

pytestmark = pytest.mark.unit


def test_agent_events_are_framework_neutral_and_serializable() -> None:
    event = AgentToolStarted(
        name="search",
        args_summary="query=Hindi OCR",
        tool_call_id="call-1",
        arguments={"query": "Hindi OCR"},
    )

    assert event_to_dict(event) == {
        "event": "AgentToolStarted",
        "name": "search",
        "args_summary": "query=Hindi OCR",
        "tool_call_id": "call-1",
        "arguments": {"query": "Hindi OCR"},
    }


def test_agent_progress_step_type_is_typed() -> None:
    event = AgentProgress(label="LLM call", detail="starting", step_type="llm")

    assert event.step_type == "llm"


def test_agent_completed_carries_answer_context_and_tool_count() -> None:
    event = AgentCompleted(answer="done", retrieved_context=["ctx"], tool_call_count=2)

    assert event.answer == "done"
    assert event.retrieved_context == ["ctx"]
    assert event.tool_call_count == 2


def test_agent_tool_finished_can_carry_persistence_metadata() -> None:
    event = AgentToolFinished(
        name="search",
        latency_ms=10.5,
        result_summary="1 public result",
        arguments={"query": "Hindi OCR"},
        result_metadata={"result_count": 1, "document_ids": ["query-kit:1"]},
    )

    assert event_to_dict(event)["result_metadata"] == {
        "result_count": 1,
        "document_ids": ["query-kit:1"],
    }
