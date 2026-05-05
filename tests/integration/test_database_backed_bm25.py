from __future__ import annotations

import os

import pytest
import pytest_asyncio

from indic_research_agent.config import AppSettings
from indic_research_agent.db.base import Base
from indic_research_agent.db.session import create_engine, create_session_factory
from indic_research_agent.retrieval.index_store import DatabaseBM25IndexStore
from indic_research_agent.services.document_service import (
    DocumentChunkInput,
    DocumentService,
)

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def session_factory():
    database_url = os.environ.get("APP_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip(
            "APP_DATABASE_URL or DATABASE_URL is required for database-backed "
            "BM25 tests"
        )

    engine = create_engine(AppSettings(database_url=database_url))
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)

    factory = create_session_factory(engine=engine)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_database_chunks_become_searchable_after_refresh(session_factory) -> None:
    async with session_factory() as session:
        document_service = DocumentService(session)
        document = await document_service.add_document(
            title="Redis cache policy",
            source_uri="test://redis-cache",
            metadata={"kind": "integration"},
            chunks=[
                DocumentChunkInput(
                    chunk_key="redis-1",
                    text=(
                        "Redis cache entries use TTL expiration for repeated searches."
                    ),
                    token_count=9,
                )
            ],
        )
        document_id = document.id
        await session.commit()

    async with session_factory() as session:
        index_store = DatabaseBM25IndexStore(DocumentService(session))
        search_service = await index_store.build_search_service()

    results = search_service.search("redis ttl repeated searches", top_k=3)

    assert results[0].document_id == str(document_id)
    assert results[0].chunk_id == "redis-1"
    assert results[0].source == "test://redis-cache"
