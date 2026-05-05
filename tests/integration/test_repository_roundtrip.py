from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy import select

from indic_research_agent.config import AppSettings
from indic_research_agent.db.base import Base
from indic_research_agent.db.session import create_engine, create_session_factory
from indic_research_agent.models import AgentResponse, ToolCall
from indic_research_agent.repositories import UserRepository
from indic_research_agent.services.document_service import (
    DocumentChunkInput,
    DocumentService,
)
from indic_research_agent.services.query_service import QueryService

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def session_factory():
    database_url = os.environ.get("APP_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip(
            "APP_DATABASE_URL or DATABASE_URL is required for repository "
            "integration tests"
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
async def test_query_document_tool_response_roundtrip(session_factory) -> None:
    async with session_factory() as session:
        user = await UserRepository(session).add(
            external_id="test-user",
            display_name="Test User",
        )
        document = await DocumentService(session).add_document(
            title="BM25",
            source_uri="test://bm25",
            metadata={"kind": "test"},
            chunks=[
                DocumentChunkInput(
                    chunk_key="chunk-1",
                    text="BM25 keyword retrieval without vector search.",
                    token_count=6,
                )
            ],
        )
        query_service = QueryService(session)
        query = await query_service.record_query(
            text="What is BM25?",
            user_id=user.id,
            session_id="session-1",
        )
        tool_call = await query_service.record_tool_call(
            query_id=query.id,
            tool_name="search",
            arguments={"query": "BM25"},
            result_summary={"document_id": str(document.id)},
            latency_ms=1.5,
        )
        response = await query_service.record_response(
            query_id=query.id,
            answer="BM25 is keyword retrieval.",
            citations=[{"document_id": str(document.id)}],
            model_metadata={"model": "test"},
        )
        await session.commit()

    async with session_factory() as session:
        tool_calls = (await session.execute(select(ToolCall))).scalars().all()
        responses = (await session.execute(select(AgentResponse))).scalars().all()

    assert tool_calls[0].id == tool_call.id
    assert tool_calls[0].tool_name == "search"
    assert responses[0].id == response.id
    assert responses[0].answer == "BM25 is keyword retrieval."
