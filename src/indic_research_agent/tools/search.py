"""Search tool."""

from __future__ import annotations

from indic_research_agent.retrieval import SearchResult, SearchService
from indic_research_agent.services.cache_service import CacheService
from indic_research_agent.services.querykit_service import QueryKitService
from indic_research_agent.tools.schemas import SearchToolInput, ToolSearchResult


class SearchTool:
    """Search local BM25 chunks and public research providers."""

    def __init__(
        self,
        search_service: SearchService,
        querykit_service: QueryKitService | None = None,
        cache_service: CacheService | None = None,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self._search_service = search_service
        self._querykit_service = querykit_service
        self._cache_service = cache_service
        self._cache_ttl_seconds = cache_ttl_seconds

    async def run(self, input_data: SearchToolInput) -> list[ToolSearchResult]:
        cache_payload = input_data.model_dump(mode="json")
        if self._cache_service is not None:
            cached = await self._cache_service.get_json("tool.search", cache_payload)
            if cached is not None:
                return [ToolSearchResult.model_validate(item) for item in cached]

        results: list[SearchResult] = []
        if input_data.source in {"local", "all"}:
            results.extend(
                self._search_service.search(input_data.query, top_k=input_data.top_k)
            )
        if input_data.source in {"research", "all"}:
            if self._querykit_service is None:
                raise ValueError("query-kit service is not configured")
            results.extend(
                await self._querykit_service.search(
                    input_data.query,
                    providers=input_data.providers,
                    limit=input_data.top_k,
                    since_year=input_data.since_year,
                )
            )
        tool_results = [
            _to_tool_result(result) for result in results[: input_data.top_k]
        ]
        if self._cache_service is not None:
            await self._cache_service.set_json(
                "tool.search",
                cache_payload,
                [result.model_dump(mode="json") for result in tool_results],
                ttl_seconds=self._cache_ttl_seconds,
            )
        return tool_results


def _to_tool_result(result: SearchResult) -> ToolSearchResult:
    return ToolSearchResult(
        document_id=result.document_id,
        chunk_id=result.chunk_id,
        score=result.score,
        title=result.title,
        source=result.source,
        snippet=result.snippet,
        metadata=dict(result.metadata),
    )
