"""End-to-end smoke query for the BM25-first agent stack."""

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
from indic_research_agent.retrieval import SearchService
from indic_research_agent.services.cache_service import CacheService
from indic_research_agent.services.document_service import (
    DocumentChunkInput,
    DocumentService,
)
from indic_research_agent.services.query_service import QueryService
from indic_research_agent.tools.fetch import FetchTool
from indic_research_agent.tools.schemas import FetchToolInput, SearchToolInput
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
        fetch_input: FetchToolInput,
        source_id: str,
    ) -> None:
        self._search_input = search_input
        self._fetch_input = fetch_input
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
        if self.calls == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fetch",
                        "args": self._fetch_input.model_dump(mode="json"),
                        "id": "smoke-fetch",
                    }
                ],
            )
        return AIMessage(
            content=(
                "BM25 lexical retrieval found the seeded smoke document, fetched "
                f"its evidence, and answered with source {self._source_id}."
            )
        )


async def run_smoke_query(
    *,
    question: str,
    settings: AppSettings | None = None,
    ensure_schema: bool = True,
    cache_ttl_seconds: int = 60,
) -> SmokeResult:
    settings = settings or AppSettings()
    run_token = f"smoke-{uuid4().hex[:10]}"
    source_id = ""

    engine = create_engine(settings)
    session_factory = create_session_factory(engine=engine)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)

    try:
        await redis.ping()
        if ensure_schema:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            document_service = DocumentService(session)
            document = await document_service.add_document(
                title=f"BM25 smoke document {run_token}",
                source_uri=f"smoke://{run_token}",
                metadata={"kind": "smoke", "run_token": run_token},
                chunks=[
                    DocumentChunkInput(
                        chunk_key="smoke-chunk-1",
                        text=(
                            f"{run_token} verifies BM25 keyword retrieval without "
                            "embeddings, query-kit-compatible search tooling, "
                            "LangGraph tool calls, Redis cache hits, and "
                            "PostgreSQL persistence."
                        ),
                        token_count=24,
                        metadata={"run_token": run_token},
                    )
                ],
            )
            query_service = QueryService(session)
            query = await query_service.record_query(
                text=question,
                session_id=run_token,
            )
            await session.commit()

        source_id = f"{document.id}#smoke-chunk-1"
        search_input = SearchToolInput(
            query=f"{question} {run_token}",
            top_k=5,
            source="local",
        )
        fetch_input = FetchToolInput(
            document_id=str(document.id),
            chunk_id="smoke-chunk-1",
            max_chars=500,
        )

        async with session_factory() as session:
            document_service = DocumentService(session)
            chunks = await document_service.list_retrieval_chunks()
            search_service = SearchService(chunks)
            cache_service = CacheService(
                redis,
                default_ttl_seconds=cache_ttl_seconds,
                metadata_repository=CacheMetadataRepository(session),
            )
            search_tool = SearchTool(
                search_service,
                cache_service=cache_service,
                cache_ttl_seconds=cache_ttl_seconds,
            )
            fetch_tool = FetchTool(
                search_service,
                cache_service=cache_service,
                cache_ttl_seconds=cache_ttl_seconds,
            )
            model = SmokeToolCallingModel(
                search_input=search_input,
                fetch_input=fetch_input,
                source_id=source_id,
            )
            graph = build_agent_graph(
                model,
                search_tool=search_tool,
                fetch_tool=fetch_tool,
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
            fetch_payload = json.loads(state["retrieved_context"][1])
            if not any(
                item["document_id"] == str(document.id) for item in search_payload
            ):
                raise RuntimeError(
                    "BM25 search did not return the seeded smoke document"
                )
            if run_token not in fetch_payload["content"]:
                raise RuntimeError("Fetch did not return seeded smoke content")

            await search_tool.run(search_input)
            await fetch_tool.run(fetch_input)

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
            await query_service.record_tool_call(
                query_id=query.id,
                tool_name="fetch",
                arguments=fetch_input.model_dump(mode="json"),
                result_summary={
                    "document_id": fetch_payload["document_id"],
                    "chunk_id": fetch_payload["chunk_id"],
                    "content_chars": len(fetch_payload["content"]),
                },
            )
            response = await query_service.record_response(
                query_id=query.id,
                answer=state["final_answer"],
                citations=[
                    {
                        "document_id": str(document.id),
                        "chunk_id": "smoke-chunk-1",
                        "source": f"smoke://{run_token}",
                    }
                ],
                model_metadata={"model": "smoke-tool-calling-model"},
            )
            await session.commit()

        async with session_factory() as session:
            cache_rows = (
                await session.execute(
                    select(CacheMetadata).where(
                        CacheMetadata.namespace.in_(["tool.search", "tool.fetch"])
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
        if cache_hits < 2:
            raise RuntimeError(
                "Expected repeated search/fetch calls to hit Redis cache"
            )
        if persisted_tool_calls < 2:
            raise RuntimeError("Expected persisted search and fetch tool calls")

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
        default="How does BM25 retrieval support this agent?",
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
        help="TTL for smoke search/fetch cache entries.",
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
