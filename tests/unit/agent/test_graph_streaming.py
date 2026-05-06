from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from indic_research_agent.agent.graph import build_agent_graph
from indic_research_agent.retrieval import DocumentChunk, SearchService
from indic_research_agent.tools.fetch import FetchTool
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
                            "query": "bm25 keyword retrieval",
                            "top_k": 1,
                            "source": "local",
                        },
                        "id": "call-search",
                    }
                ],
            )
        return AIMessage(content="BM25 works with streaming progress.")


@pytest.mark.asyncio
async def test_graph_astream_emits_custom_progress_events() -> None:
    search_service = SearchService(
        [
            DocumentChunk(
                document_id="bm25",
                chunk_id="bm25-1",
                title="BM25",
                source="test://bm25",
                text="BM25 keyword retrieval ranks chunks without embeddings.",
            )
        ]
    )
    graph = build_agent_graph(
        StreamingFakeModel(),
        search_tool=SearchTool(search_service),
        fetch_tool=FetchTool(search_service),
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
    assert final_answer == "BM25 works with streaming progress."
