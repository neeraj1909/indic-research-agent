from __future__ import annotations

import os

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage
from sqlalchemy import select

from indic_research_agent.config import AppSettings
from indic_research_agent.db.base import Base
from indic_research_agent.db.session import create_engine, create_session_factory
from indic_research_agent.models import AgentResponse, ToolCall, UserQuery
from indic_research_agent.services.agent_service import AgentService

pytestmark = pytest.mark.integration


class PersistingFakeGraph:
    async def astream(self, input_state, **kwargs):
        yield (
            "custom",
            {
                "event": "agent.tool.start",
                "name": "search",
                "tool_call_id": "call-1",
                "args": {"query": "bm25"},
                "args_summary": "query=bm25",
            },
        )
        yield (
            "custom",
            {
                "event": "agent.tool.end",
                "name": "search",
                "tool_call_id": "call-1",
                "latency_ms": 4.0,
                "result_summary": "1 result(s): BM25",
                "result_metadata": {"result_count": 1, "document_ids": ["doc-1"]},
            },
        )
        yield (
            "updates",
            {
                "llm": {
                    "messages": [AIMessage(content="BM25 answer")],
                    "retrieved_context": ["context"],
                    "tool_call_count": 1,
                    "final_answer": "BM25 answer",
                }
            },
        )


@pytest_asyncio.fixture
async def session_factory():
    database_url = os.environ.get("APP_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip(
            "APP_DATABASE_URL or DATABASE_URL is required for persistence tests"
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
async def test_ui_style_agent_stream_records_query_tool_and_response(
    session_factory,
) -> None:
    service = AgentService(
        PersistingFakeGraph(),
        session_factory=session_factory,
        persist_queries=True,
    )

    events = [
        event
        async for event in service.stream_answer(
            "How does BM25 help?",
            session_id="chainlit-thread-1",
            user_identifier="test",
        )
    ]

    assert events[-1].answer == "BM25 answer"
    async with session_factory() as session:
        queries = (await session.execute(select(UserQuery))).scalars().all()
        tool_calls = (await session.execute(select(ToolCall))).scalars().all()
        responses = (await session.execute(select(AgentResponse))).scalars().all()

    assert len(queries) == 1
    assert queries[0].session_id == "chainlit-thread-1"
    assert queries[0].text == "How does BM25 help?"
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "search"
    assert tool_calls[0].arguments_json == {"query": "bm25"}
    assert tool_calls[0].result_summary_json["result_count"] == 1
    assert len(responses) == 1
    assert responses[0].answer == "BM25 answer"
