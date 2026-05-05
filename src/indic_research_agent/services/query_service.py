"""Query lifecycle service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from indic_research_agent.models import AgentResponse, ToolCall, UserQuery
from indic_research_agent.repositories import (
    AgentResponseRepository,
    QueryRepository,
    ToolCallRepository,
)


class QueryService:
    def __init__(self, session: AsyncSession) -> None:
        self._queries = QueryRepository(session)
        self._tool_calls = ToolCallRepository(session)
        self._responses = AgentResponseRepository(session)

    async def record_query(
        self,
        *,
        text: str,
        user_id: UUID | None = None,
        session_id: str | None = None,
        rewritten_text: str | None = None,
    ) -> UserQuery:
        return await self._queries.add(
            text=text,
            user_id=user_id,
            session_id=session_id,
            rewritten_text=rewritten_text,
        )

    async def record_tool_call(
        self,
        *,
        query_id: UUID,
        tool_name: str,
        arguments: Mapping[str, Any],
        result_summary: Mapping[str, Any],
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> ToolCall:
        return await self._tool_calls.add(
            query_id=query_id,
            tool_name=tool_name,
            arguments=arguments,
            result_summary=result_summary,
            latency_ms=latency_ms,
            error=error,
        )

    async def record_response(
        self,
        *,
        query_id: UUID,
        answer: str,
        citations: Sequence[Mapping[str, Any]] | None = None,
        model_metadata: Mapping[str, Any] | None = None,
    ) -> AgentResponse:
        return await self._responses.add(
            query_id=query_id,
            answer=answer,
            citations=citations,
            model_metadata=model_metadata,
        )
