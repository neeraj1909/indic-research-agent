"""User repository."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from indic_research_agent.models import User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(
        self,
        *,
        external_id: str | None = None,
        display_name: str | None = None,
    ) -> User:
        user = User(external_id=external_id, display_name=display_name)
        self._session.add(user)
        await self._session.flush()
        return user

    async def get(self, user_id: UUID) -> User | None:
        return await self._session.get(User, user_id)
