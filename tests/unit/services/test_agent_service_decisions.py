from __future__ import annotations

import asyncio

import pytest

from indic_research_agent.agent.decision import DecisionRoute
from indic_research_agent.config import AppSettings
from indic_research_agent.services.agent_events import AgentCompleted, AgentProgress
from indic_research_agent.services.agent_service import AgentService

pytestmark = pytest.mark.unit


class Graph:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_input = None

    async def astream(self, input_state, **kwargs):
        self.calls += 1
        self.seen_input = input_state
        yield (
            "updates",
            {
                "answer": "graph answer",
                "retrieved_context": ["source"],
                "tool_call_count": 1,
                "final_answer": "graph answer",
            },
        )


def response(route: DecisionRoute) -> dict[str, object]:
    return {
        "model": "gpt-6-luna",
        "answers": [{"type": "choice", "name": "route", "choice": route.value}],
    }


class TransformationModel:
    def __init__(self, answer: str = "Transformed supplied text") -> None:
        self.answer = answer
        self.messages = []

    async def ainvoke(self, messages):
        self.messages.append(messages)
        from langchain_core.messages import AIMessage

        return AIMessage(content=self.answer)


@pytest.mark.asyncio
async def test_off_mode_makes_no_decision_call() -> None:
    calls = 0

    async def provider(_request):
        nonlocal calls
        calls += 1
        return response(DecisionRoute.CLARIFY)

    graph = Graph()
    events = [
        event
        async for event in AgentService(
            graph,
            decisions_provider=provider,
            settings=AppSettings(_env_file=None),
        ).stream_answer("Question")
    ]

    assert calls == 0
    assert graph.calls == 1
    assert events[-1].answer == "graph answer"


@pytest.mark.asyncio
async def test_active_clarify_route_changes_behavior_without_graph_call() -> None:
    calls = []

    async def provider(request):
        calls.append(request)
        return response(DecisionRoute.CLARIFY)

    graph = Graph()
    events = [
        event
        async for event in AgentService(
            graph,
            decisions_provider=provider,
            settings=AppSettings(_env_file=None, decisions_mode="active"),
        ).stream_answer("Question")
    ]

    assert len(calls) == 1
    assert graph.calls == 0
    assert isinstance(events[-1], AgentCompleted)
    assert "clarify" in events[-1].answer.lower()


@pytest.mark.asyncio
async def test_shadow_calls_once_but_keeps_graph_authoritative() -> None:
    calls = []

    async def provider(request):
        calls.append(request)
        return response(DecisionRoute.CLARIFY)

    graph = Graph()
    events = [
        event
        async for event in AgentService(
            graph,
            decisions_provider=provider,
            settings=AppSettings(_env_file=None, decisions_mode="shadow"),
        ).stream_answer("Question")
    ]

    assert len(calls) == 1
    assert graph.calls == 1
    assert events[-1].answer == "graph answer"


@pytest.mark.asyncio
async def test_active_research_route_preserves_existing_graph() -> None:
    async def provider(_request):
        return response(DecisionRoute.RESEARCH)

    graph = Graph()
    events = [
        event
        async for event in AgentService(
            graph,
            decisions_provider=provider,
            settings=AppSettings(_env_file=None, decisions_mode="active"),
        ).stream_answer("Find sources")
    ]

    assert graph.calls == 1
    assert events[-1].answer == "graph answer"


@pytest.mark.asyncio
async def test_active_provider_failure_preserves_existing_graph() -> None:
    async def provider(_request):
        raise RuntimeError("provider must not escape")

    graph = Graph()
    events = [
        event
        async for event in AgentService(
            graph,
            decisions_provider=provider,
            settings=AppSettings(_env_file=None, decisions_mode="active"),
        ).stream_answer("Find sources")
    ]

    assert graph.calls == 1
    assert events[-1].answer == "graph answer"


@pytest.mark.asyncio
async def test_default_decision_timeout_allows_normal_provider_and_timeout_falls_back():
    async def provider(_request):
        await asyncio.sleep(0.01)
        return response(DecisionRoute.RESEARCH)

    graph = Graph()
    events = [
        event
        async for event in AgentService(
            graph,
            decisions_provider=provider,
            settings=AppSettings(_env_file=None, decisions_mode="active"),
        ).stream_answer("Find sources")
    ]
    assert graph.calls == 1
    assert events[-1].answer == "graph answer"

    async def hanging_provider(_request):
        await asyncio.sleep(1)
        return response(DecisionRoute.RESEARCH)

    graph = Graph()
    events = [
        event
        async for event in AgentService(
            graph,
            decisions_provider=hanging_provider,
            settings=AppSettings(
                _env_file=None,
                decisions_mode="active",
                decisions_timeout_seconds=0.001,
            ),
        ).stream_answer("Find sources")
    ]
    assert graph.calls == 1
    assert any(
        isinstance(event, AgentProgress)
        and event.detail == "status=fallback route=legacy"
        for event in events
    )


@pytest.mark.asyncio
async def test_active_supplied_text_uses_no_search_transformation_path_and_persists(
    monkeypatch,
) -> None:
    async def provider(_request):
        return response(DecisionRoute.SUPPLIED_TEXT)

    model = TransformationModel("Edited text")
    graph = Graph()
    persisted = []
    service = AgentService(
        graph,
        transformation_model=model,
        decisions_provider=provider,
        settings=AppSettings(_env_file=None, decisions_mode="active"),
    )

    async def record_response(**kwargs):
        persisted.append(kwargs)

    monkeypatch.setattr(service, "_record_response", record_response)

    async def record_query(**_kwargs):
        return 1

    monkeypatch.setattr(service, "_record_query", record_query)

    events = [
        event
        async for event in service.stream_answer(
            "Copyedit this exact text: teh report is ready.", persist=True
        )
    ]

    assert graph.calls == 0
    assert model.messages[0][-1].content == (
        "Copyedit this exact text: teh report is ready."
    )
    assert events[-1].answer == "Edited text"
    assert any(
        isinstance(event, AgentProgress)
        and event.detail == "status=ok route=supplied_text"
        for event in events
    )
    assert persisted[-1]["event"].answer == "Edited text"


@pytest.mark.asyncio
async def test_supplied_text_transformation_failure_is_safe() -> None:
    async def provider(_request):
        return response(DecisionRoute.SUPPLIED_TEXT)

    class BrokenModel:
        async def ainvoke(self, _messages):
            raise RuntimeError("provider detail must not escape")

    events = [
        event
        async for event in AgentService(
            Graph(),
            transformation_model=BrokenModel(),
            decisions_provider=provider,
            settings=AppSettings(_env_file=None, decisions_mode="active"),
        ).stream_answer("Translate this text")
    ]

    assert events[-1].message == "text transformation failed"
