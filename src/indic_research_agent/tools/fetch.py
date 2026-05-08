"""Fetch tool."""

from __future__ import annotations

from indic_research_agent.retrieval import SearchService
from indic_research_agent.services.cache_service import CacheService
from indic_research_agent.services.phoenix_tracing import set_span_output, trace_span
from indic_research_agent.tools.schemas import FetchResult, FetchToolInput


class FetchTool:
    """Fetch a bounded local document chunk by id."""

    def __init__(
        self,
        search_service: SearchService,
        cache_service: CacheService | None = None,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self._search_service = search_service
        self._cache_service = cache_service
        self._cache_ttl_seconds = cache_ttl_seconds

    async def run(self, input_data: FetchToolInput) -> FetchResult:
        cache_payload = input_data.model_dump(mode="json")
        if self._cache_service is not None:
            cached = await self._cache_service.get_json("tool.fetch", cache_payload)
            if cached is not None:
                return FetchResult.model_validate(cached)

        with trace_span(
            "fetch.local_bm25_chunk",
            kind="RETRIEVER",
            input_value=cache_payload,
            attributes={"retrieval.source": "local"},
        ) as fetch_span:
            chunk = self._search_service.fetch(
                input_data.document_id,
                chunk_id=input_data.chunk_id,
            )
            content = chunk.text[: input_data.max_chars]
            set_span_output(
                fetch_span,
                {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "content": content,
                    "content_chars": len(content),
                    "metadata": dict(chunk.metadata),
                },
            )
        result = FetchResult(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            source=chunk.source,
            content=content,
            metadata=dict(chunk.metadata),
        )
        if self._cache_service is not None:
            await self._cache_service.set_json(
                "tool.fetch",
                cache_payload,
                result.model_dump(mode="json"),
                ttl_seconds=self._cache_ttl_seconds,
            )
        return result
