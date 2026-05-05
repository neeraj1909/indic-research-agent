"""Agent response repository."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from indic_research_agent.models import AgentResponse


class AgentResponseRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(
        self,
        *,
        query_id: UUID,
        answer: str,
        citations: Sequence[Mapping[str, Any]] | None = None,
        model_metadata: Mapping[str, Any] | None = None,
    ) -> AgentResponse:
        response = AgentResponse(
            query_id=query_id,
            answer=answer,
            citations_json=[dict(citation) for citation in citations or []],
            model_metadata_json=dict(model_metadata or {}),
        )
        self._session.add(response)
        await self._session.flush()
        return response
