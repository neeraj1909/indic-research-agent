"""Async database session factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from indic_research_agent.config import AppSettings, get_settings


def create_engine(settings: AppSettings | None = None) -> AsyncEngine:
    settings = settings or get_settings()
    return create_async_engine(settings.database_url, pool_pre_ping=True)


def create_session_factory(
    engine: AsyncEngine | None = None,
    settings: AppSettings | None = None,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=engine or create_engine(settings),
        expire_on_commit=False,
    )


@asynccontextmanager
async def session_scope(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> AsyncIterator[AsyncSession]:
    factory = session_factory or create_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
