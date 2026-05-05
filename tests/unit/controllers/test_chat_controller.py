from __future__ import annotations

import pytest

from indic_research_agent.controllers import ChatController
from indic_research_agent.services.agent_service import AgentAnswer

pytestmark = pytest.mark.unit


class FakeAgentService:
    async def answer(self, question: str) -> AgentAnswer:
        return AgentAnswer(
            answer=f"answer:{question}",
            retrieved_context=[],
            tool_call_count=0,
        )


@pytest.mark.asyncio
async def test_chat_controller_delegates_to_agent_service() -> None:
    controller = ChatController(FakeAgentService())

    result = await controller.handle_message("hello")

    assert result.answer == "answer:hello"
