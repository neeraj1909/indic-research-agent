from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from indic_research_agent.config import AppSettings
from indic_research_agent.services import agent_service as agent_service_module
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
                "args": {"query": "Hindi OCR"},
                "args_summary": "query=Hindi OCR",
            },
        )
        yield (
            "custom",
            {
                "event": "agent.tool.end",
                "name": "search",
                "tool_call_id": "call-1",
                "latency_ms": 12.5,
                "result_summary": "1 result(s): Hindi OCR source",
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
        yield ("messages", (AIMessage(content="Hindi OCR "), {}))
        yield ("messages", (AIMessage(content="works."), {}))
        yield (
            "updates",
            {
                "llm": {
                    "messages": [AIMessage(content="Hindi OCR works.")],
                    "retrieved_context": ["ctx"],
                    "tool_call_count": 1,
                    "final_answer": "Hindi OCR works.",
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
    assert finished.arguments == {"query": "Hindi OCR"}
    assert [
        getattr(event, "text", None) for event in events if hasattr(event, "text")
    ] == [
        "Hindi OCR ",
        "works.",
    ]
    completed = events[-1]
    assert isinstance(completed, AgentCompleted)
    assert completed.answer == "Hindi OCR works."
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

    assert answer.answer == "Hindi OCR works."
    assert answer.tool_call_count == 1


def test_default_graph_uses_injected_settings(monkeypatch) -> None:
    settings = AppSettings(_env_file=None, litellm_model="injected/model")
    seen: dict[str, object] = {}
    sentinel_model = object()
    sentinel_graph = object()

    def fake_create_chat_model(received_settings):
        seen["settings"] = received_settings
        return sentinel_model

    def fake_build_agent_graph(model, *, search_tool):
        seen["model"] = model
        seen["search_tool"] = search_tool
        return sentinel_graph

    monkeypatch.setattr(
        agent_service_module, "create_chat_model", fake_create_chat_model
    )
    monkeypatch.setattr(
        agent_service_module, "build_agent_graph", fake_build_agent_graph
    )

    service = AgentService(settings=settings)

    assert service._settings is settings
    assert service._graph is sentinel_graph
    assert seen["settings"] is settings
    assert seen["model"] is sentinel_model


def test_default_graph_without_settings_uses_global_settings(monkeypatch) -> None:
    settings = AppSettings(_env_file=None, litellm_model="default/model")
    seen: dict[str, object] = {}

    monkeypatch.setattr(agent_service_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        agent_service_module,
        "create_chat_model",
        lambda received_settings: seen.setdefault("settings", received_settings),
    )
    monkeypatch.setattr(
        agent_service_module,
        "build_agent_graph",
        lambda model, *, search_tool: seen.setdefault("model", model),
    )

    graph = agent_service_module.create_default_graph()

    assert graph is seen["model"]
    assert seen["settings"] is settings
