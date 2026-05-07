"""Chat workflow controller."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from uuid import uuid4

from indic_research_agent.services.agent_events import AgentCompleted, AgentStreamEvent
from indic_research_agent.services.agent_service import AgentAnswer, AgentService
from indic_research_agent.services.chat_history import (
    ChatTurn,
    append_exchange,
    cap_history,
    normalize_history,
)


class ChatController:
    """Thin controller between UI adapters and the agent service."""

    def __init__(
        self,
        agent_service: AgentService | None = None,
        *,
        message_history: Sequence[ChatTurn | dict[str, object]] | None = None,
        app_session_id: str | None = None,
        chainlit_thread_id: str | None = None,
        user_identifier: str | None = None,
        persist_queries: bool = False,
    ) -> None:
        self._agent_service = agent_service or AgentService(
            persist_queries=persist_queries
        )
        self.message_history = normalize_history(list(message_history or []))
        self.app_session_id = app_session_id or f"chat-{uuid4().hex}"
        self.chainlit_thread_id = chainlit_thread_id
        self.user_identifier = user_identifier
        self._persist_queries = persist_queries

    async def handle_message(self, content: str) -> AgentAnswer:
        return await self._agent_service.answer(
            content,
            history=self.message_history,
            session_id=self.session_id,
            user_identifier=self.user_identifier,
        )

    async def stream_message(self, content: str) -> AsyncIterator[AgentStreamEvent]:
        completed: AgentCompleted | None = None
        async for event in self._agent_service.stream_answer(
            content,
            history=self.message_history,
            session_id=self.session_id,
            user_identifier=self.user_identifier,
            persist=self._persist_queries,
        ):
            if isinstance(event, AgentCompleted):
                completed = event
            yield event
        if completed is not None:
            self.message_history = append_exchange(
                self.message_history,
                user_content=content,
                assistant_content=completed.answer,
            )

    @property
    def session_id(self) -> str:
        return self.chainlit_thread_id or self.app_session_id

    def set_history(self, history: Sequence[ChatTurn | dict[str, object]]) -> None:
        self.message_history = cap_history(normalize_history(list(history)))
