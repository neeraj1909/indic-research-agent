from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from indic_research_agent.agent.graph import build_agent_graph
from indic_research_agent.retrieval import DocumentChunk, SearchService
from indic_research_agent.retrieval.search_service import SearchResult
from indic_research_agent.tools.fetch import FetchTool
from indic_research_agent.tools.search import SearchTool

pytestmark = pytest.mark.unit


class FakeToolCallingModel:
    def __init__(self) -> None:
        self.calls = 0
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
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
                            "top_k": 2,
                            "source": "all",
                        },
                        "id": "call-search",
                    }
                ],
            )
        if self.calls == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fetch",
                        "args": {
                            "document_id": "bm25",
                            "chunk_id": "bm25-1",
                            "max_chars": 120,
                        },
                        "id": "call-fetch",
                    }
                ],
            )
        return AIMessage(content="BM25 retrieval works with fetched evidence.")


class BudgetExhaustionModel:
    def __init__(self) -> None:
        self.bound_calls = 0
        self.unbound_calls = 0

    def bind_tools(self, tools):
        return BoundBudgetExhaustionModel(self)

    async def ainvoke(self, messages):
        self.unbound_calls += 1
        assert "tool-call budget is exhausted" in messages[-1].content
        return AIMessage(
            content=(
                "Use the retrieved BM25 result for the final answer. [S1]\n\n"
                "Sources:\n- [S1] BM25, local corpus, bm25/bm25-1."
            )
        )


class BoundBudgetExhaustionModel:
    def __init__(self, parent: BudgetExhaustionModel) -> None:
        self.parent = parent

    async def ainvoke(self, messages):
        self.parent.bound_calls += 1
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


class FakeQueryKitService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def search(self, query, *, providers=None, limit=5, since_year=None):
        self.calls.append(
            {
                "query": query,
                "providers": providers,
                "limit": limit,
                "since_year": since_year,
            }
        )
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
        ]


@pytest.mark.asyncio
async def test_graph_executes_search_then_fetch_then_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span_outputs: list[object] = []

    def capture_span_output(span, value):
        span_outputs.append(value)

    monkeypatch.setattr(
        "indic_research_agent.agent.graph.set_span_output",
        capture_span_output,
    )

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
    model = FakeToolCallingModel()
    querykit_service = FakeQueryKitService()
    graph = build_agent_graph(
        model,
        search_tool=SearchTool(search_service, querykit_service),
        fetch_tool=FetchTool(search_service),
    )

    result = await graph.ainvoke(
        {
            "messages": [],
            "tool_call_count": 0,
            "retrieved_context": [],
            "final_answer": None,
        }
    )

    assert model.bound_tools is not None
    assert model.calls == 3
    assert result["final_answer"] == "BM25 retrieval works with fetched evidence."
    assert result["tool_call_count"] == 2
    assert len(result["retrieved_context"]) == 2
    search_payload = json.loads(result["retrieved_context"][0])
    fetch_payload = json.loads(result["retrieved_context"][1])
    assert querykit_service.calls
    assert {item["document_id"] for item in search_payload} == {
        "bm25",
        "query-kit:research-1",
    }
    assert fetch_payload["content"].startswith("BM25 keyword retrieval")
    search_span_output = next(
        output
        for output in span_outputs
        if isinstance(output, dict)
        and output.get("summary", "").startswith("2 result(s)")
    )
    assert search_span_output["metadata"]["result_count"] == 2
    assert len(search_span_output["results"]) == 2
    assert search_span_output["results"][0]["snippet"].startswith(
        "BM25 keyword retrieval"
    )
    assert search_span_output["results"][0]["citation_id"] == "S1"


@pytest.mark.asyncio
async def test_graph_forces_final_answer_when_tool_budget_is_exhausted() -> None:
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
    model = BudgetExhaustionModel()
    graph = build_agent_graph(
        model,
        search_tool=SearchTool(search_service),
        fetch_tool=FetchTool(search_service),
        max_tool_calls=1,
    )

    result = await graph.ainvoke(
        {
            "messages": [],
            "tool_call_count": 0,
            "retrieved_context": [],
            "final_answer": None,
        }
    )

    assert model.bound_calls == 1
    assert model.unbound_calls == 1
    assert result["tool_call_count"] == 1
    assert result["final_answer"].startswith("Use the retrieved BM25 result")
    assert result["final_answer"].strip()
