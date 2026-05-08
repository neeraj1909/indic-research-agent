"""Search tool."""

from __future__ import annotations

import logging
import time

from indic_research_agent.retrieval import SearchResult, SearchService
from indic_research_agent.services.cache_service import CacheService
from indic_research_agent.services.phoenix_tracing import set_span_output, trace_span
from indic_research_agent.services.progress_events import emit_agent_progress
from indic_research_agent.services.querykit_service import QueryKitService
from indic_research_agent.tools.schemas import SearchToolInput, ToolSearchResult

logger = logging.getLogger(__name__)


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
        search_summary = _search_summary(input_data)
        if self._cache_service is not None:
            cached = await self._cache_service.get_json("tool.search", cache_payload)
            if cached is not None:
                emit_agent_progress(
                    label="Search cache hit",
                    detail=(
                        f"{search_summary}; returned {len(cached)} cached result(s)."
                    ),
                    step_type="retrieval",
                )
                return [ToolSearchResult.model_validate(item) for item in cached]

        logger.info("search.tool.start %s", search_summary)
        emit_agent_progress(
            label="Search started",
            detail=search_summary,
            step_type="retrieval",
        )
        results: list[SearchResult] = []
        if input_data.source in {"local", "all"}:
            start = time.perf_counter()
            emit_agent_progress(
                label="Search: local BM25",
                detail=(
                    f"Running local keyword retrieval for {_quote(input_data.query)}."
                ),
                step_type="retrieval",
            )
            with trace_span(
                "search.local_bm25",
                kind="RETRIEVER",
                input_value={"query": input_data.query, "top_k": input_data.top_k},
                attributes={"retrieval.source": "local"},
            ) as local_span:
                local_results = self._search_service.search(
                    input_data.query, top_k=input_data.top_k
                )
                set_span_output(
                    local_span,
                    [_search_result_trace_payload(result) for result in local_results],
                )
            elapsed_ms = (time.perf_counter() - start) * 1000
            results.extend(local_results)
            logger.info(
                "search.local_bm25.end query=%r results=%s elapsed_ms=%.0f",
                input_data.query,
                len(local_results),
                elapsed_ms,
            )
            emit_agent_progress(
                label="Search: local BM25 complete",
                detail=f"{len(local_results)} result(s) in {elapsed_ms:.0f} ms.",
                step_type="retrieval",
            )
        if input_data.source in {"research", "all"}:
            if self._querykit_service is None:
                raise ValueError("query-kit service is not configured")
            start = time.perf_counter()
            provider_detail = (
                ",".join(input_data.providers)
                if input_data.providers
                else "configured providers"
            )
            emit_agent_progress(
                label="Search: public research",
                detail=(
                    f"Running query-kit for {_quote(input_data.query)} "
                    f"against {provider_detail}."
                ),
                step_type="retrieval",
            )
            try:
                with trace_span(
                    "search.query_kit",
                    kind="RETRIEVER",
                    input_value={
                        "query": input_data.query,
                        "providers": input_data.providers,
                        "limit": input_data.top_k,
                        "since_year": input_data.since_year,
                    },
                    attributes={"retrieval.source": "query-kit"},
                ) as research_span:
                    research_results = await self._querykit_service.search(
                        input_data.query,
                        providers=input_data.providers,
                        limit=input_data.top_k,
                        since_year=input_data.since_year,
                    )
                    if not research_results:
                        for fallback_query in _provider_friendly_queries(
                            input_data.query
                        ):
                            emit_agent_progress(
                                label="Search: public research retry",
                                detail=(
                                    "No public-provider results for "
                                    f"{_quote(input_data.query)}; retrying "
                                    f"with {_quote(fallback_query)}."
                                ),
                                step_type="retrieval",
                            )
                            research_results = await self._querykit_service.search(
                                fallback_query,
                                providers=input_data.providers,
                                limit=input_data.top_k,
                                since_year=input_data.since_year,
                            )
                            if research_results:
                                logger.info(
                                    (
                                        "search.query_kit.retry_success "
                                        "original_query=%r retry_query=%r results=%s"
                                    ),
                                    input_data.query,
                                    fallback_query,
                                    len(research_results),
                                )
                                break
                    set_span_output(
                        research_span,
                        [
                            _search_result_trace_payload(result)
                            for result in research_results
                        ],
                    )
            except Exception as exc:
                logger.warning("search.querykit_failed", exc_info=True)
                emit_agent_progress(
                    label="Search: public research failed",
                    detail=(
                        f"{type(exc).__name__}: {_truncate(str(exc), 240)}. "
                        f"Continuing with {len(results)} local result(s)."
                    ),
                    step_type="retrieval",
                )
                research_results = []
            elapsed_ms = (time.perf_counter() - start) * 1000
            results.extend(research_results)
            logger.info(
                "search.query_kit.end query=%r results=%s elapsed_ms=%.0f",
                input_data.query,
                len(research_results),
                elapsed_ms,
            )
            emit_agent_progress(
                label="Search: public research complete",
                detail=f"{len(research_results)} result(s) in {elapsed_ms:.0f} ms.",
                step_type="retrieval",
            )
        tool_results = [
            _to_tool_result(result) for result in results[: input_data.top_k]
        ]
        logger.info("search.tool.end results=%s %s", len(tool_results), search_summary)
        emit_agent_progress(
            label="Search complete",
            detail=f"Returning {len(tool_results)} result(s) for {search_summary}.",
            step_type="retrieval",
        )
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


def _search_summary(input_data: SearchToolInput) -> str:
    providers = ",".join(input_data.providers or []) or "configured"
    since = f", since_year={input_data.since_year}" if input_data.since_year else ""
    return (
        f"query={_quote(input_data.query)}, top_k={input_data.top_k}, "
        f"source={input_data.source}, providers={providers}{since}"
    )


def _provider_friendly_queries(query: str) -> list[str]:
    lower_query = query.casefold()
    fallbacks: list[str] = []

    def add(candidate: str) -> None:
        normalized_candidate = " ".join(candidate.split())
        if not normalized_candidate:
            return
        if normalized_candidate.casefold() == " ".join(query.split()).casefold():
            return
        if normalized_candidate not in fallbacks:
            fallbacks.append(normalized_candidate)

    if "hindi" in lower_query and "ocr" in lower_query:
        add("Hindi OCR")
    if "devanagari" in lower_query and "ocr" in lower_query:
        add("Devanagari OCR")
    if "indic" in lower_query and "ocr" in lower_query:
        add("Indic OCR")
    if "dataset" in lower_query:
        if "hindi" in lower_query and "ocr" in lower_query:
            add("Hindi OCR dataset")
        if "devanagari" in lower_query and "ocr" in lower_query:
            add("Devanagari OCR dataset")
    return fallbacks[:3]


def _search_result_trace_payload(result: SearchResult) -> dict[str, object]:
    return {
        "document_id": result.document_id,
        "chunk_id": result.chunk_id,
        "title": result.title,
        "source": result.source,
        "score": result.score,
        "snippet": result.snippet,
        "metadata": dict(result.metadata),
    }


def _quote(value: str) -> str:
    return f'"{_truncate(value, 160)}"'


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 3]}..."
