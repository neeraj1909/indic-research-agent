"""query-kit integration service."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from query_cli.application.services import search_research_async
from query_cli.bootstrap import get_search_providers
from query_cli.domain.errors import ProviderSearchError

from indic_research_agent.config import AppSettings, get_settings
from indic_research_agent.retrieval import SearchResult
from indic_research_agent.services.cache_service import CacheService
from indic_research_agent.services.phoenix_tracing import set_span_output, trace_span
from indic_research_agent.services.progress_events import emit_agent_progress

ProviderFactory = Callable[..., Sequence[Any]]
SearchFunction = Callable[..., Awaitable[Sequence[Any]]]
logger = logging.getLogger(__name__)
FALLBACK_PROVIDER_IDS = ("arxiv", "semantic-scholar", "openreview", "pubmed", "acl")


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

        start = time.perf_counter()
        timeout_seconds = max(0.1, self._settings.query_kit_timeout_seconds)
        logger.info(
            (
                "query-kit.search.start query=%r providers=%s limit=%s "
                "since_year=%s timeout_seconds=%.1f"
            ),
            query,
            provider_ids,
            limit,
            since_year,
            timeout_seconds,
        )
        emit_agent_progress(
            label="Query-kit search started",
            detail=(
                f"providers={','.join(provider_ids)}; limit={limit}; "
                f"timeout={timeout_seconds:.0f}s"
            ),
            step_type="retrieval",
        )
        try:
            raw_results = await asyncio.wait_for(
                self._search_with_provider_fallbacks(
                    query=query,
                    provider_ids=provider_ids,
                    limit=limit,
                    since_year=since_year,
                    deadline=time.monotonic() + timeout_seconds,
                ),
                timeout=timeout_seconds + 0.5,
            )
        except TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "query-kit search timed out query=%r providers=%s timeout_seconds=%.1f",
                query,
                provider_ids,
                timeout_seconds,
            )
            emit_agent_progress(
                label="Query-kit search timed out",
                detail=(
                    f"No public-provider results returned within "
                    f"{timeout_seconds:.0f}s ({elapsed_ms:.0f} ms elapsed)."
                ),
                step_type="retrieval",
            )
            return []
        except ProviderSearchError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("query-kit search failed: %s", exc)
            emit_agent_progress(
                label="Query-kit search failed",
                detail=(
                    f"{exc}. Continuing without public-provider results "
                    f"after {elapsed_ms:.0f} ms."
                ),
                step_type="retrieval",
            )
            return []
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "query-kit.search.end query=%r raw_results=%s elapsed_ms=%.0f",
            query,
            len(raw_results),
            elapsed_ms,
        )
        emit_agent_progress(
            label="Query-kit search complete",
            detail=f"{len(raw_results)} raw result(s) in {elapsed_ms:.0f} ms.",
            step_type="retrieval",
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
        deadline: float,
    ) -> Sequence[Any]:
        try:
            return await self._search_provider_ids(
                query=query,
                provider_ids=provider_ids,
                limit=limit,
                since_year=since_year,
                deadline=deadline,
            )
        except TimeoutError:
            raise
        except ProviderSearchError as exc:
            fallback_provider_ids = _fallback_provider_ids(provider_ids)
            if len(fallback_provider_ids) <= 1:
                raise
            logger.warning(
                "query-kit combined search failed; retrying providers individually: %s",
                exc,
            )
            emit_agent_progress(
                label="Query-kit fallback",
                detail=(
                    "Combined provider search failed; retrying individually: "
                    + ",".join(fallback_provider_ids)
                ),
                step_type="retrieval",
            )

        raw_results: list[Any] = []
        for provider_id in fallback_provider_ids:
            if _remaining_seconds(deadline) <= 0:
                emit_agent_progress(
                    label="Query-kit fallback budget exhausted",
                    detail=(
                        f"Stopped before provider={provider_id}; "
                        "returning partial results."
                    ),
                    step_type="retrieval",
                )
                break
            try:
                provider_results = await self._search_provider_ids(
                    query=query,
                    provider_ids=[provider_id],
                    limit=limit,
                    since_year=since_year,
                    deadline=deadline,
                )
            except TimeoutError:
                raise
            except ProviderSearchError as exc:
                logger.warning(
                    "query-kit provider failed provider=%s: %s", provider_id, exc
                )
                emit_agent_progress(
                    label="Query-kit provider failed",
                    detail=f"provider={provider_id}: {exc}",
                    step_type="retrieval",
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
        deadline: float,
    ) -> Sequence[Any]:
        provider_timeout = min(
            self._settings.query_kit_timeout_seconds,
            max(0.1, _remaining_seconds(deadline)),
        )
        provider_instances = self._provider_factory(
            list(provider_ids),
            timeout=provider_timeout,
            environ=self._environ,
        )
        provider_names = [_provider_id(provider) for provider in provider_instances]
        remaining = max(0.1, _remaining_seconds(deadline))
        logger.info(
            (
                "query-kit.providers.start query=%r providers=%s "
                "limit=%s since_year=%s budget_seconds=%.1f"
            ),
            query,
            provider_names,
            limit,
            since_year,
            remaining,
        )
        emit_agent_progress(
            label="Query-kit providers running",
            detail=f"providers={','.join(provider_names)}; budget={remaining:.0f}s",
            step_type="retrieval",
        )

        if len(provider_instances) == 1:
            results = await self._search_single_provider(
                query=query,
                provider=provider_instances[0],
                limit=limit,
                since_year=since_year,
                deadline=deadline,
            )
            logger.info(
                "query-kit.providers.end providers=%s results=%s failures=0",
                provider_names,
                len(results),
            )
            emit_agent_progress(
                label="Query-kit providers complete",
                detail=f"providers={provider_names[0]}; results={len(results)}",
                step_type="retrieval",
            )
            return results

        tasks = {
            asyncio.create_task(
                self._search_single_provider(
                    query=query,
                    provider=provider,
                    limit=limit,
                    since_year=since_year,
                    deadline=deadline,
                )
            ): _provider_id(provider)
            for provider in provider_instances
        }
        start = time.perf_counter()
        done, pending = await asyncio.wait(
            tasks,
            timeout=max(0.1, _remaining_seconds(deadline)),
        )

        provider_result_sets: list[Sequence[Any]] = []
        failures: list[str] = []
        for task in done:
            provider_name = tasks[task]
            try:
                provider_results = task.result()
            except TimeoutError:
                failures.append(f"{provider_name}: timed out")
            except ProviderSearchError as exc:
                failures.append(f"{provider_name}: {exc}")
            except Exception as exc:
                failures.append(f"{provider_name}: {type(exc).__name__}: {exc}")
            else:
                provider_result_sets.append(provider_results)

        for task in pending:
            provider_name = tasks[task]
            task.cancel()
            failures.append(f"{provider_name}: timed out")
            logger.warning("query-kit.provider.timeout provider=%s", provider_name)
            emit_agent_progress(
                label="Query-kit provider timed out",
                detail=f"provider={provider_name}; budget exhausted",
                step_type="retrieval",
            )

        results = _merge_raw_result_sets(provider_result_sets, limit=limit)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            (
                "query-kit.providers.end providers=%s results=%s failures=%s "
                "elapsed_ms=%.0f"
            ),
            provider_names,
            len(results),
            len(failures),
            elapsed_ms,
        )
        emit_agent_progress(
            label="Query-kit providers complete",
            detail=(
                f"providers={','.join(provider_names)}; results={len(results)}; "
                f"failures={len(failures)}; elapsed={elapsed_ms:.0f} ms"
            ),
            step_type="retrieval",
        )
        if not provider_result_sets and failures:
            raise ProviderSearchError("all", "; ".join(failures))
        return results

    async def _search_single_provider(
        self,
        *,
        query: str,
        provider: Any,
        limit: int,
        since_year: int | None,
        deadline: float,
    ) -> Sequence[Any]:
        provider_name = _provider_id(provider)
        remaining = max(0.1, _remaining_seconds(deadline))
        logger.info(
            (
                "query-kit.provider.start provider=%s query=%r "
                "limit=%s since_year=%s budget_seconds=%.1f"
            ),
            provider_name,
            query,
            limit,
            since_year,
            remaining,
        )
        emit_agent_progress(
            label="Query-kit provider running",
            detail=f"provider={provider_name}; budget={remaining:.0f}s",
            step_type="retrieval",
        )
        start = time.perf_counter()
        with trace_span(
            f"query-kit.provider.{provider_name}",
            kind="RETRIEVER",
            input_value={
                "query": query,
                "provider": provider_name,
                "limit": limit,
                "since_year": since_year,
                "budget_seconds": remaining,
            },
            attributes={
                "query_kit.provider": provider_name,
                "query_kit.limit": limit,
            },
        ) as provider_span:
            try:
                results = await asyncio.wait_for(
                    self._search_function(
                        query,
                        [provider],
                        limit=limit,
                        since_year=since_year,
                    ),
                    timeout=remaining,
                )
            except TimeoutError:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.warning(
                    "query-kit.provider.timeout provider=%s elapsed_ms=%.0f",
                    provider_name,
                    elapsed_ms,
                )
                emit_agent_progress(
                    label="Query-kit provider timed out",
                    detail=f"provider={provider_name} after {elapsed_ms:.0f} ms",
                    step_type="retrieval",
                )
                raise
            except ProviderSearchError as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.warning(
                    "query-kit.provider.failed provider=%s elapsed_ms=%.0f error=%s",
                    provider_name,
                    elapsed_ms,
                    exc,
                )
                emit_agent_progress(
                    label="Query-kit provider failed",
                    detail=f"provider={provider_name}: {exc}",
                    step_type="retrieval",
                )
                raise
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.warning(
                    (
                        "query-kit.provider.failed provider=%s elapsed_ms=%.0f "
                        "error=%s: %s"
                    ),
                    provider_name,
                    elapsed_ms,
                    type(exc).__name__,
                    exc,
                )
                emit_agent_progress(
                    label="Query-kit provider failed",
                    detail=f"provider={provider_name}: {type(exc).__name__}: {exc}",
                    step_type="retrieval",
                )
                raise ProviderSearchError(
                    provider_name,
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            elapsed_ms = (time.perf_counter() - start) * 1000
            set_span_output(
                provider_span,
                {
                    "result_count": len(results),
                    "results": [
                        _raw_result_trace_payload(result) for result in results
                    ],
                    "elapsed_ms": elapsed_ms,
                },
            )
        logger.info(
            "query-kit.provider.end provider=%s results=%s elapsed_ms=%.0f",
            provider_name,
            len(results),
            elapsed_ms,
        )
        emit_agent_progress(
            label="Query-kit provider complete",
            detail=(
                f"provider={provider_name}; results={len(results)}; "
                f"elapsed={elapsed_ms:.0f} ms"
            ),
            step_type="retrieval",
        )
        return results

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


def _remaining_seconds(deadline: float) -> float:
    return deadline - time.monotonic()


def _provider_id(provider: Any) -> str:
    return str(getattr(provider, "provider_id", provider))


def _merge_raw_result_sets(
    provider_result_sets: Sequence[Sequence[Any]],
    *,
    limit: int,
) -> list[Any]:
    results: list[Any] = []
    seen: set[tuple[str, str]] = set()
    max_results = max(
        (len(result_set) for result_set in provider_result_sets), default=0
    )
    for index in range(max_results):
        for provider_results in provider_result_sets:
            if index >= len(provider_results):
                continue
            result = provider_results[index]
            dedupe_key = getattr(result, "dedupe_key", None)
            if dedupe_key is None:
                dedupe_key = (
                    str(getattr(result, "source", "")),
                    str(getattr(result, "url", getattr(result, "title", ""))),
                )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            results.append(result)
            if len(results) >= limit:
                return results
    return results


def _raw_result_trace_payload(result: Any) -> dict[str, Any]:
    return {
        "title": str(getattr(result, "title", "")),
        "url": str(getattr(result, "url", "")),
        "source": str(getattr(result, "source", "")),
        "authors": list(getattr(result, "authors", ()) or ()),
        "year": getattr(result, "year", None),
        "venue": getattr(result, "venue", None),
        "abstract": getattr(result, "abstract", None),
    }


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
