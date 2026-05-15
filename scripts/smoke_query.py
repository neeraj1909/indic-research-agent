"""End-to-end smoke query for the public research search agent stack."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from redis.asyncio import Redis
from sqlalchemy import select

from indic_research_agent.agent.graph import build_agent_graph
from indic_research_agent.config import AppSettings
from indic_research_agent.db.base import Base
from indic_research_agent.db.session import create_engine, create_session_factory
from indic_research_agent.models import CacheMetadata, ToolCall
from indic_research_agent.repositories import CacheMetadataRepository
from indic_research_agent.services.cache_service import CacheService
from indic_research_agent.services.query_service import QueryService
from indic_research_agent.services.search_results import SearchResult
from indic_research_agent.tools.schemas import SearchToolInput
from indic_research_agent.tools.search import SearchTool


@dataclass(frozen=True)
class SmokeResult:
    answer: str
    source_ids: list[str]
    tool_call_count: int
    persisted_tool_calls: int
    cache_hits: int
    cache_namespaces: list[str]
    query_id: str
    response_id: str


class SmokeToolCallingModel:
    """Deterministic model used to test LangGraph tool wiring offline."""

    def __init__(
        self,
        *,
        search_input: SearchToolInput,
        source_id: str,
    ) -> None:
        self._search_input = search_input
        self._source_id = source_id
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search",
                        "args": self._search_input.model_dump(mode="json"),
                        "id": "smoke-search",
                    }
                ],
            )
        return AIMessage(
            content=(
                "Public research search found the deterministic smoke source "
                f"{self._source_id} and answered with provider-backed evidence."
            )
        )


class SmokeQueryKitService:
    """Deterministic public-research provider stand-in for smoke validation."""

    def __init__(self, *, run_token: str, source_id: str) -> None:
        self._run_token = run_token
        self._source_id = source_id
        self.calls: list[dict[str, object]] = []

    async def search(self, query, *, providers=None, limit=5, since_year=None):
        self.calls.append(
            {
                "query": query,
                "providers": providers,
                "limit": limit,
                "since_year": since_year,
            }
        )
        return [
            SearchResult(
                document_id=self._source_id,
                chunk_id="abstract",
                score=1.0,
                title="Deterministic Hindi OCR public research smoke source",
                source=f"https://example.test/research/{self._run_token}",
                snippet=(
                    f"{self._run_token} verifies public research search, "
                    "query-kit-compatible tool calls, Redis cache hits, and "
                    "PostgreSQL persistence without local document search."
                ),
                metadata={
                    "provider": "smoke-public-provider",
                    "year": 2026,
                    "run_token": self._run_token,
                },
            )
        ][:limit]


async def run_smoke_query(
    *,
    question: str,
    settings: AppSettings | None = None,
    ensure_schema: bool = True,
    cache_ttl_seconds: int = 60,
) -> SmokeResult:
    settings = settings or AppSettings()
    run_token = f"smoke-{uuid4().hex[:10]}"
    source_id = f"query-kit:smoke:{run_token}"

    engine = create_engine(settings)
    session_factory = create_session_factory(engine=engine)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)

    try:
        await redis.ping()
        if ensure_schema:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            query_service = QueryService(session)
            query = await query_service.record_query(
                text=question,
                session_id=run_token,
            )
            await session.commit()

        search_input = SearchToolInput(
            query=f"{question} {run_token}",
            top_k=5,
            providers=["semantic-scholar", "pubmed", "arxiv-web"],
            since_year=2020,
        )

        async with session_factory() as session:
            cache_service = CacheService(
                redis,
                default_ttl_seconds=cache_ttl_seconds,
                metadata_repository=CacheMetadataRepository(session),
            )
            querykit_service = SmokeQueryKitService(
                run_token=run_token,
                source_id=source_id,
            )
            search_tool = SearchTool(
                querykit_service,
                cache_service=cache_service,
                cache_ttl_seconds=cache_ttl_seconds,
            )
            model = SmokeToolCallingModel(
                search_input=search_input,
                source_id=source_id,
            )
            graph = build_agent_graph(
                model,
                search_tool=search_tool,
            )
            state = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=question)],
                    "tool_call_count": 0,
                    "retrieved_context": [],
                    "final_answer": None,
                }
            )

            search_payload = json.loads(state["retrieved_context"][0])
            if not any(item["document_id"] == source_id for item in search_payload):
                raise RuntimeError(
                    "Public research search did not return the smoke source"
                )
            if run_token not in search_payload[0]["snippet"]:
                raise RuntimeError("Search did not return smoke source content")

            await search_tool.run(search_input)

            query_service = QueryService(session)
            await query_service.record_tool_call(
                query_id=query.id,
                tool_name="search",
                arguments=search_input.model_dump(mode="json"),
                result_summary={
                    "result_count": len(search_payload),
                    "document_ids": [item["document_id"] for item in search_payload],
                },
            )
            response = await query_service.record_response(
                query_id=query.id,
                answer=state["final_answer"],
                citations=[
                    {
                        "document_id": source_id,
                        "source": f"https://example.test/research/{run_token}",
                    }
                ],
                model_metadata={"model": "smoke-tool-calling-model"},
            )
            await session.commit()

        async with session_factory() as session:
            cache_rows = (
                await session.execute(
                    select(CacheMetadata).where(
                        CacheMetadata.namespace == "tool.search"
                    )
                )
            ).scalars()
            cache_metadata = list(cache_rows)
            tool_calls = (
                await session.execute(
                    select(ToolCall).where(ToolCall.query_id == query.id)
                )
            ).scalars()
            persisted_tool_calls = len(list(tool_calls))

        cache_hits = sum(row.hit_count for row in cache_metadata)
        if cache_hits < 1:
            raise RuntimeError("Expected repeated search call to hit Redis cache")
        if persisted_tool_calls < 1:
            raise RuntimeError("Expected persisted search tool call")

        return SmokeResult(
            answer=state["final_answer"],
            source_ids=[source_id],
            tool_call_count=state["tool_call_count"],
            persisted_tool_calls=persisted_tool_calls,
            cache_hits=cache_hits,
            cache_namespaces=sorted({row.namespace for row in cache_metadata}),
            query_id=str(query.id),
            response_id=str(response.id),
        )
    finally:
        await redis.aclose()
        await engine.dispose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default="Find public research on Hindi OCR.",
        help="Question to send through the deterministic smoke graph.",
    )
    parser.add_argument(
        "--skip-schema-create",
        action="store_true",
        help="Assume migrations have already created the database schema.",
    )
    parser.add_argument(
        "--cache-ttl-seconds",
        type=int,
        default=60,
        help="TTL for smoke search cache entries.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    result = await run_smoke_query(
        question=args.question,
        ensure_schema=not args.skip_schema_create,
        cache_ttl_seconds=args.cache_ttl_seconds,
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
