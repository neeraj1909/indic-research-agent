from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from indic_research_agent.agent.graph import build_agent_graph
from indic_research_agent.services.search_results import SearchResult
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
                            "query": "Hindi OCR public research",
                            "top_k": 2,
                            "providers": ["arxiv"],
                        },
                        "id": "call-search",
                    }
                ],
            )
        return AIMessage(content="Public research search works with evidence.")


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
                "Use the retrieved public research result for the final answer. "
                "[S1]\n\nSources:\n- [S1] Query-kit research result, arxiv."
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
                        "query": "Hindi OCR public research",
                        "top_k": 1,
                        "providers": ["arxiv"],
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
async def test_graph_executes_public_search_then_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span_outputs: list[object] = []

    def capture_span_output(span, value):
        span_outputs.append(value)

    monkeypatch.setattr(
        "indic_research_agent.agent.graph.set_span_output",
        capture_span_output,
    )

    model = FakeToolCallingModel()
    querykit_service = FakeQueryKitService()
    graph = build_agent_graph(
        model,
        search_tool=SearchTool(querykit_service),
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
    assert [tool.name for tool in model.bound_tools] == ["search"]
    assert model.calls == 2
    assert result["final_answer"] == "Public research search works with evidence."
    assert result["tool_call_count"] == 1
    assert len(result["retrieved_context"]) == 1
    search_payload = json.loads(result["retrieved_context"][0])
    assert querykit_service.calls == [
        {
            "query": "Hindi OCR public research",
            "providers": ["arxiv"],
            "limit": 2,
            "since_year": None,
        }
    ]
    assert {item["document_id"] for item in search_payload} == {"query-kit:research-1"}
    search_span_output = next(
        output
        for output in span_outputs
        if isinstance(output, dict)
        and output.get("summary", "").startswith("1 result(s)")
    )
    assert search_span_output["metadata"]["result_count"] == 1
    assert len(search_span_output["results"]) == 1
    assert search_span_output["results"][0]["snippet"].startswith("Public research")
    assert search_span_output["results"][0]["citation_id"] == "S1"


@pytest.mark.asyncio
async def test_graph_forces_final_answer_when_tool_budget_is_exhausted() -> None:
    model = BudgetExhaustionModel()
    graph = build_agent_graph(
        model,
        search_tool=SearchTool(FakeQueryKitService()),
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
    assert result["final_answer"].startswith("Use the retrieved public research")
    assert result["final_answer"].strip()
