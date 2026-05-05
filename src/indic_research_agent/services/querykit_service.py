"""query-kit integration service."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from query_cli.application.services import search_research_async
from query_cli.bootstrap import get_search_providers
from query_cli.domain.errors import ProviderSearchError

from indic_research_agent.config import AppSettings, get_settings
from indic_research_agent.retrieval import SearchResult
from indic_research_agent.services.cache_service import CacheService

ProviderFactory = Callable[..., Sequence[Any]]
SearchFunction = Callable[..., Awaitable[Sequence[Any]]]
logger = logging.getLogger(__name__)
FALLBACK_PROVIDER_IDS = ("acl", "arxiv", "pubmed", "semantic-scholar", "openreview")


class QueryKitService:
    """Async wrapper around query-kit's Python API."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        provider_factory: ProviderFactory = get_search_providers,
        search_function: SearchFunction = search_research_async,
        environ: Mapping[str, str] | None = None,
        cache_service: CacheService | None = None,
        cache_ttl_seconds: int = 900,
    ) -> None:
        self._settings = settings or get_settings()
        self._provider_factory = provider_factory
        self._search_function = search_function
        self._environ = environ if environ is not None else os.environ
        self._cache_service = cache_service
        self._cache_ttl_seconds = cache_ttl_seconds

    async def search(
        self,
        query: str,
        *,
        providers: Sequence[str] | None = None,
        limit: int = 5,
        since_year: int | None = None,
    ) -> list[SearchResult]:
        provider_ids = list(providers or self._settings.query_kit_provider_ids)
        if limit <= 0:
            raise ValueError("limit must be greater than 0")
        if not query.strip():
            return []

        cache_payload = {
            "query": query,
            "providers": provider_ids,
            "limit": limit,
            "since_year": since_year,
            "timeout": self._settings.query_kit_timeout_seconds,
        }
        if self._cache_service is not None:
            cached = await self._cache_service.get_json(
                "query-kit.search", cache_payload
            )
            if cached is not None:
                return [SearchResult(**item) for item in cached]

        raw_results = await self._search_with_provider_fallbacks(
            query=query,
            provider_ids=provider_ids,
            limit=limit,
            since_year=since_year,
        )
        results = [self._to_search_result(result) for result in raw_results]
        if self._cache_service is not None:
            await self._cache_service.set_json(
                "query-kit.search",
                cache_payload,
                [_search_result_to_dict(result) for result in results],
                ttl_seconds=self._cache_ttl_seconds,
            )
        return results

    async def _search_with_provider_fallbacks(
        self,
        *,
        query: str,
        provider_ids: Sequence[str],
        limit: int,
        since_year: int | None,
    ) -> Sequence[Any]:
        try:
            return await self._search_provider_ids(
                query=query,
                provider_ids=provider_ids,
                limit=limit,
                since_year=since_year,
            )
        except ProviderSearchError as exc:
            fallback_provider_ids = _fallback_provider_ids(provider_ids)
            if len(fallback_provider_ids) <= 1:
                raise
            logger.warning(
                "query-kit combined search failed; retrying providers individually: %s",
                exc,
            )

        raw_results: list[Any] = []
        for provider_id in fallback_provider_ids:
            try:
                provider_results = await self._search_provider_ids(
                    query=query,
                    provider_ids=[provider_id],
                    limit=limit,
                    since_year=since_year,
                )
            except ProviderSearchError as exc:
                logger.warning(
                    "query-kit provider failed provider=%s: %s", provider_id, exc
                )
                continue
            raw_results.extend(provider_results)
            if len(raw_results) >= limit:
                break
        return raw_results[:limit]

    async def _search_provider_ids(
        self,
        *,
        query: str,
        provider_ids: Sequence[str],
        limit: int,
        since_year: int | None,
    ) -> Sequence[Any]:
        provider_instances = self._provider_factory(
            list(provider_ids),
            timeout=self._settings.query_kit_timeout_seconds,
            environ=self._environ,
        )
        return await self._search_function(
            query,
            provider_instances,
            limit=limit,
            since_year=since_year,
        )

    def _to_search_result(self, result: Any) -> SearchResult:
        title = str(result.title)
        url = str(result.url)
        source = str(result.source)
        abstract = getattr(result, "abstract", None)
        snippet = str(abstract or title)
        result_id = _stable_result_id(title=title, url=url, source=source)
        metadata = {
            "result_type": "query-kit",
            "url": url,
            "provider": source,
            "authors": list(getattr(result, "authors", ()) or ()),
            "year": getattr(result, "year", None),
            "venue": getattr(result, "venue", None),
        }
        return SearchResult(
            document_id=result_id,
            chunk_id="abstract",
            score=1.0,
            title=title,
            source=url,
            snippet=snippet,
            metadata=metadata,
        )


def _stable_result_id(*, title: str, url: str, source: str) -> str:
    digest = hashlib.sha256(f"{source}\n{title}\n{url}".encode()).hexdigest()[:16]
    return f"query-kit:{digest}"


def _fallback_provider_ids(provider_ids: Sequence[str]) -> list[str]:
    if any(provider_id == "all" for provider_id in provider_ids):
        return list(FALLBACK_PROVIDER_IDS)
    return list(dict.fromkeys(provider_ids))


def _search_result_to_dict(result: SearchResult) -> dict[str, Any]:
    return {
        "document_id": result.document_id,
        "chunk_id": result.chunk_id,
        "score": result.score,
        "snippet": result.snippet,
        "title": result.title,
        "source": result.source,
        "metadata": dict(result.metadata),
    }
