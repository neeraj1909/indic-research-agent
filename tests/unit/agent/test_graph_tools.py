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


class BatchToolCallingModel:
    def __init__(self, tool_calls: list[dict[str, object]]) -> None:
        self.tool_calls = tool_calls
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=self.tool_calls)
        return AIMessage(content="Final answer after tool errors.")


class TwoSearchModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls < 3:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search",
                        "args": {"query": f"query-{self.calls}", "top_k": 1},
                        "id": f"call-search-{self.calls}",
                    }
                ],
            )
        return AIMessage(content="Final answer with both sources.")


class DistinctFakeQueryKitService(FakeQueryKitService):
    async def search(self, query, *, providers=None, limit=5, since_year=None):
        self.calls.append(
            {
                "query": query,
                "providers": providers,
                "limit": limit,
                "since_year": since_year,
            }
        )
        document_id = f"query-kit:research-{len(self.calls)}"
        return [
            SearchResult(
                document_id=document_id,
                chunk_id="abstract",
                score=1.0,
                title=f"Result {document_id}",
                source=f"https://example.test/{document_id}",
                snippet="Public research result from query-kit.",
                metadata={"provider": "fake"},
            )
        ]


class FailingQueryKitService(FakeQueryKitService):
    async def search(self, query, *, providers=None, limit=5, since_year=None):
        raise RuntimeError("secret-token https://private.example/api")


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


@pytest.mark.asyncio
async def test_graph_caps_batch_to_remaining_budget_and_reports_each_call() -> None:
    model = BatchToolCallingModel(
        [
            {
                "name": "search",
                "args": {"query": "first", "top_k": 1},
                "id": "call-first",
            },
            {
                "name": "search",
                "args": {"query": "second", "top_k": 1},
                "id": "call-second",
            },
            {
                "name": "search",
                "args": {"query": "third", "top_k": 1},
                "id": "call-third",
            },
        ]
    )
    querykit_service = FakeQueryKitService()
    graph = build_agent_graph(
        model,
        search_tool=SearchTool(querykit_service),
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

    tool_messages = [
        message for message in result["messages"] if message.type == "tool"
    ]
    assert len(querykit_service.calls) == 1
    assert [message.tool_call_id for message in tool_messages] == [
        "call-first",
        "call-second",
        "call-third",
    ]
    assert "not executed" in tool_messages[1].content
    assert "not executed" in tool_messages[2].content
    assert result["tool_call_count"] == 1
    assert model.calls == 2


@pytest.mark.asyncio
async def test_graph_reports_invalid_tool_and_args_without_raising() -> None:
    model = BatchToolCallingModel(
        [
            {"name": "not-search", "args": {}, "id": "call-invalid-tool"},
            {"name": "search", "args": {"top_k": 1}, "id": "call-invalid-args"},
        ]
    )
    graph = build_agent_graph(
        model,
        search_tool=SearchTool(FakeQueryKitService()),
        max_tool_calls=2,
    )

    result = await graph.ainvoke(
        {
            "messages": [],
            "tool_call_count": 0,
            "retrieved_context": [],
            "final_answer": None,
        }
    )

    tool_messages = [
        message for message in result["messages"] if message.type == "tool"
    ]
    assert "unknown tool" in tool_messages[0].content
    assert "invalid arguments" in tool_messages[1].content
    assert result["final_answer"] == "Final answer after tool errors."


@pytest.mark.asyncio
async def test_graph_admits_unknown_first_call_and_rejects_valid_excess_call() -> None:
    model = BatchToolCallingModel(
        [
            {"name": "not-search", "args": {}, "id": "call-unknown"},
            {
                "name": "search",
                "args": {"query": "must not run", "top_k": 1},
                "id": "call-excess",
            },
        ]
    )
    querykit_service = FakeQueryKitService()
    graph = build_agent_graph(
        model,
        search_tool=SearchTool(querykit_service),
        max_tool_calls=1,
    )

    result = await graph.ainvoke(
        {"messages": [], "tool_call_count": 0, "retrieved_context": []}
    )

    tool_messages = [
        message for message in result["messages"] if message.type == "tool"
    ]
    assert [message.tool_call_id for message in tool_messages] == [
        "call-unknown",
        "call-excess",
    ]
    assert "unknown tool" in tool_messages[0].content
    assert "budget exhausted" in tool_messages[1].content
    assert querykit_service.calls == []
    assert result["tool_call_count"] == 1


@pytest.mark.asyncio
async def test_graph_rejects_unknown_excess_call_after_valid_search() -> None:
    model = BatchToolCallingModel(
        [
            {
                "name": "search",
                "args": {"query": "runs once", "top_k": 1},
                "id": "call-search",
            },
            {"name": "not-search", "args": {}, "id": "call-excess-unknown"},
        ]
    )
    querykit_service = FakeQueryKitService()
    graph = build_agent_graph(
        model,
        search_tool=SearchTool(querykit_service),
        max_tool_calls=1,
    )

    result = await graph.ainvoke(
        {"messages": [], "tool_call_count": 0, "retrieved_context": []}
    )

    tool_messages = [
        message for message in result["messages"] if message.type == "tool"
    ]
    assert len(querykit_service.calls) == 1
    assert tool_messages[1].tool_call_id == "call-excess-unknown"
    assert "budget exhausted" in tool_messages[1].content


@pytest.mark.asyncio
async def test_graph_sanitizes_runtime_search_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BatchToolCallingModel(
        [{"name": "search", "args": {"query": "fails"}, "id": "call-fails"}]
    )
    search_tool = SearchTool(FailingQueryKitService())

    async def fail_at_graph_boundary(_validated_args):
        raise RuntimeError("secret-token https://private.example/api")

    monkeypatch.setattr(search_tool, "run", fail_at_graph_boundary)
    graph = build_agent_graph(
        model,
        search_tool=search_tool,
        max_tool_calls=1,
    )

    result = await graph.ainvoke(
        {"messages": [], "tool_call_count": 0, "retrieved_context": []}
    )

    tool_message = next(
        message for message in result["messages"] if message.type == "tool"
    )
    assert "invocation failed" in tool_message.content
    assert "secret-token" not in tool_message.content
    assert "private.example" not in tool_message.content


def test_graph_rejects_negative_tool_budget() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_agent_graph(
            FakeToolCallingModel(),
            search_tool=SearchTool(FakeQueryKitService()),
            max_tool_calls=-1,
        )


@pytest.mark.asyncio
async def test_graph_assigns_unique_citations_across_searches() -> None:
    querykit_service = DistinctFakeQueryKitService()
    graph = build_agent_graph(
        TwoSearchModel(),
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

    contexts = [json.loads(payload) for payload in result["retrieved_context"]]
    assert [items[0]["citation_id"] for items in contexts] == ["S1", "S2"]
    assert [items[0]["metadata"]["citation_id"] for items in contexts] == [
        "S1",
        "S2",
    ]
