from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from indic_research_agent.services.agent_events import (
    AgentCompleted,
    AgentProgress,
    AgentRunStarted,
    AgentToolFinished,
    AgentToolStarted,
)
from indic_research_agent.services.agent_service import AgentService
from indic_research_agent.services.chat_history import ChatTurn

pytestmark = pytest.mark.unit


class FakeStreamingGraph:
    def __init__(self) -> None:
        self.seen_input = None

    async def astream(self, input_state, **kwargs):
        self.seen_input = input_state
        yield (
            "custom",
            {
                "event": "agent.tool.start",
                "name": "search",
                "tool_call_id": "call-1",
                "args": {"query": "bm25"},
                "args_summary": "query=bm25",
            },
        )
        yield (
            "custom",
            {
                "event": "agent.tool.end",
                "name": "search",
                "tool_call_id": "call-1",
                "latency_ms": 12.5,
                "result_summary": "1 result(s): BM25",
                "result_metadata": {"result_count": 1},
            },
        )
        yield (
            "custom",
            {
                "event": "agent.progress",
                "label": "Query-kit providers running",
                "detail": "providers=arxiv; budget=20s",
                "step_type": "retrieval",
            },
        )
        yield ("messages", (AIMessage(content="BM25 "), {}))
        yield ("messages", (AIMessage(content="works."), {}))
        yield (
            "updates",
            {
                "llm": {
                    "messages": [AIMessage(content="BM25 works.")],
                    "retrieved_context": ["ctx"],
                    "tool_call_count": 1,
                    "final_answer": "BM25 works.",
                }
            },
        )


@pytest.mark.asyncio
async def test_stream_answer_emits_ordered_progress_tokens_and_completion() -> None:
    graph = FakeStreamingGraph()
    service = AgentService(graph)

    events = [
        event
        async for event in service.stream_answer(
            "How?",
            history=[ChatTurn(role="user", content="Earlier question")],
            session_id="thread-1",
            user_identifier="test",
        )
    ]

    assert isinstance(events[0], AgentRunStarted)
    assert isinstance(events[1], AgentProgress)
    assert any(isinstance(event, AgentToolStarted) for event in events)
    assert any(
        isinstance(event, AgentProgress)
        and event.label == "Query-kit providers running"
        and event.step_type == "retrieval"
        for event in events
    )
    finished = next(event for event in events if isinstance(event, AgentToolFinished))
    assert finished.arguments == {"query": "bm25"}
    assert [
        getattr(event, "text", None) for event in events if hasattr(event, "text")
    ] == [
        "BM25 ",
        "works.",
    ]
    completed = events[-1]
    assert isinstance(completed, AgentCompleted)
    assert completed.answer == "BM25 works."
    assert completed.retrieved_context == ["ctx"]
    assert completed.tool_call_count == 1
    assert [message.content for message in graph.seen_input["messages"]] == [
        "Earlier question",
        "How?",
    ]


@pytest.mark.asyncio
async def test_answer_collects_final_completion_from_stream() -> None:
    service = AgentService(FakeStreamingGraph())

    answer = await service.answer("How?")

    assert answer.answer == "BM25 works."
    assert answer.tool_call_count == 1
