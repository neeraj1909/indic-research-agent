"""Query repository."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from indic_research_agent.models import UserQuery


class QueryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(
        self,
        *,
        text: str,
        user_id: UUID | None = None,
        session_id: str | None = None,
        rewritten_text: str | None = None,
    ) -> UserQuery:
        query = UserQuery(
            user_id=user_id,
            session_id=session_id,
            text=text,
            rewritten_text=rewritten_text,
        )
        self._session.add(query)
        await self._session.flush()
        return query

    async def get(self, query_id: UUID) -> UserQuery | None:
        return await self._session.get(UserQuery, query_id)
