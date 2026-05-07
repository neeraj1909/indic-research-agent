from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from indic_research_agent.services.agent_events import (
    AgentCompleted,
    AgentProgress,
    AgentToken,
    AgentToolFinished,
    AgentToolStarted,
)
from indic_research_agent.services.agent_service import AgentService

pytestmark = pytest.mark.e2e


class SmokeStreamingGraph:
    async def astream(self, input_state, **kwargs):
        yield ("custom", {"event": "agent.llm.start", "messages": 1})
        yield (
            "custom",
            {
                "event": "agent.tool.start",
                "name": "search",
                "tool_call_id": "search-1",
                "args": {"query": "bm25"},
                "args_summary": "query=bm25",
            },
        )
        yield (
            "custom",
            {
                "event": "agent.tool.end",
                "name": "search",
                "tool_call_id": "search-1",
                "latency_ms": 1,
                "result_summary": "1 result(s): BM25",
            },
        )
        yield ("messages", (AIMessage(content="streamed "), {}))
        yield ("messages", (AIMessage(content="answer"), {}))
        yield (
            "updates",
            {
                "llm": {
                    "messages": [AIMessage(content="streamed answer")],
                    "final_answer": "streamed answer",
                    "retrieved_context": ["ctx"],
                    "tool_call_count": 1,
                }
            },
        )


@pytest.mark.asyncio
async def test_service_stream_smoke_has_progress_tokens_and_completion() -> None:
    service = AgentService(SmokeStreamingGraph())

    events = [event async for event in service.stream_answer("Explain BM25")]

    assert any(isinstance(event, AgentProgress) for event in events)
    assert any(isinstance(event, AgentToolStarted) for event in events)
    assert any(isinstance(event, AgentToolFinished) for event in events)
    assert "".join(event.text for event in events if isinstance(event, AgentToken)) == (
        "streamed answer"
    )
    completed = events[-1]
    assert isinstance(completed, AgentCompleted)
    assert completed.answer == "streamed answer"
