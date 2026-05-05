"""Chat workflow controller."""

from __future__ import annotations

from indic_research_agent.services.agent_service import AgentAnswer, AgentService


class ChatController:
    """Thin controller between UI adapters and the agent service."""

    def __init__(self, agent_service: AgentService | None = None) -> None:
        self._agent_service = agent_service or AgentService()

    async def handle_message(self, content: str) -> AgentAnswer:
        return await self._agent_service.answer(content)
