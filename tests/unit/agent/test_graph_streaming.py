from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from indic_research_agent.agent.graph import build_agent_graph
from indic_research_agent.services.search_results import SearchResult
from indic_research_agent.tools.search import SearchTool

pytestmark = pytest.mark.unit


class StreamingFakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search",
                        "args": {
                            "query": "Hindi OCR public research",
                            "top_k": 1,
                        },
                        "id": "call-search",
                    }
                ],
            )
        return AIMessage(content="Public research works with streaming progress.")


class FakeQueryKitService:
    async def search(self, query, *, providers=None, limit=5, since_year=None):
        return [
            SearchResult(
                document_id="query-kit:research-1",
                chunk_id="abstract",
                score=1.0,
                title="Query-kit research result",
                source="https://example.test/research",
                snippet="Public research result from query-kit.",
                metadata={"provider": "fake"},
            )
        ][:limit]


@pytest.mark.asyncio
async def test_graph_astream_emits_custom_progress_events() -> None:
    graph = build_agent_graph(
        StreamingFakeModel(),
        search_tool=SearchTool(FakeQueryKitService()),
    )

    custom_events = []
    final_answer = None
    async for mode, payload in graph.astream(
        {
            "messages": [],
            "tool_call_count": 0,
            "retrieved_context": [],
            "final_answer": None,
        },
        stream_mode=["updates", "custom"],
    ):
        if mode == "custom":
            custom_events.append(payload)
        elif mode == "updates" and "llm" in payload:
            final_answer = payload["llm"].get("final_answer") or final_answer

    event_names = [event["event"] for event in custom_events]
    assert "agent.llm.start" in event_names
    assert "agent.tool.start" in event_names
    assert "agent.tool.end" in event_names
    assert "agent.answer.finalized" in event_names
    assert final_answer == "Public research works with streaming progress."
