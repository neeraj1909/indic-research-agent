from __future__ import annotations

import pytest

from indic_research_agent.controllers import ChatController
from indic_research_agent.services.agent_events import AgentCompleted, AgentRunStarted
from indic_research_agent.services.agent_service import AgentAnswer
from indic_research_agent.services.chat_history import ChatTurn

pytestmark = pytest.mark.unit


class FakeAgentService:
    def __init__(self) -> None:
        self.seen = []

    async def answer(self, question, **kwargs) -> AgentAnswer:
        self.seen.append((question, kwargs))
        return AgentAnswer(
            answer=f"answer:{question}",
            retrieved_context=[],
            tool_call_count=0,
        )

    async def stream_answer(self, question, **kwargs):
        self.seen.append((question, kwargs))
        yield AgentRunStarted(
            question=question,
            session_id=kwargs.get("session_id"),
            user_identifier=kwargs.get("user_identifier"),
        )
        yield AgentCompleted(
            answer=f"answer:{question}",
            retrieved_context=[],
            tool_call_count=0,
        )


@pytest.mark.asyncio
async def test_chat_controller_delegates_to_agent_service() -> None:
    service = FakeAgentService()
    controller = ChatController(service)

    result = await controller.handle_message("hello")

    assert result.answer == "answer:hello"
    assert service.seen[0][1]["history"] == []


@pytest.mark.asyncio
async def test_chat_controller_streams_with_session_and_updates_history() -> None:
    service = FakeAgentService()
    controller = ChatController(
        service,
        message_history=[ChatTurn(role="user", content="earlier")],
        app_session_id="app-session",
        chainlit_thread_id="thread-1",
        user_identifier="test",
        persist_queries=True,
    )

    events = [event async for event in controller.stream_message("hello")]

    assert isinstance(events[0], AgentRunStarted)
    assert service.seen[0][1]["session_id"] == "thread-1"
    assert service.seen[0][1]["user_identifier"] == "test"
    assert service.seen[0][1]["persist"] is True
    assert controller.message_history[-2:] == [
        ChatTurn(role="user", content="hello"),
        ChatTurn(role="assistant", content="answer:hello"),
    ]
