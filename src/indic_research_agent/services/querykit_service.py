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
from indic_research_agent.services.phoenix_tracing import (
    set_span_attributes,
    set_span_output,
    trace_span,
)
from indic_research_agent.services.progress_events import emit_agent_progress

ProviderFactory = Callable[..., Sequence[Any]]
SearchFunction = Callable[..., Awaitable[Sequence[Any]]]
logger = logging.getLogger(__name__)
FALLBACK_PROVIDER_IDS = (
    "semantic-scholar",
    "semantic-scholar-web",
    "pubmed",
    "arxiv-web",
    "arxiv",
    "openreview",
    "acl",
)
KNOWN_PROVIDER_IDS = frozenset((*FALLBACK_PROVIDER_IDS, "all"))
PROVIDER_ALIASES = {
    "semanticscholar": "semantic-scholar",
    "semantic_scholar": "semantic-scholar",
    "semantic scholar": "semantic-scholar",
    "s2": "semantic-scholar",
    "arxiv": "arxiv-web",
    "arxiv-web": "arxiv-web",
    "arxiv_web": "arxiv-web",
}


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
        requested_provider_ids = list(
            providers or self._settings.query_kit_provider_ids
        )
        provider_ids = _normalize_provider_ids(
            requested_provider_ids,
            default_provider_ids=self._settings.query_kit_provider_ids,
        )
        if provider_ids != requested_provider_ids:
            logger.info(
                "query-kit.providers.normalized requested=%s resolved=%s",
                requested_provider_ids,
                provider_ids,
            )
            _progress(
                "Query-kit providers normalized",
                (
                    f"requested={','.join(requested_provider_ids)}; "
                    f"using={','.join(provider_ids)}"
                ),
            )
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
        if (
            self._cache_service is not None
            and (
                cached := await self._cache_service.get_json(
                    "query-kit.search", cache_payload
                )
            )
            is not None
        ):
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
        _progress(
            "Query-kit search started",
            (
                f"providers={','.join(provider_ids)}; limit={limit}; "
                f"timeout={timeout_seconds:.0f}s"
            ),
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
            _progress(
                "Query-kit search timed out",
                (
                    f"No public-provider results returned within "
                    f"{timeout_seconds:.0f}s ({elapsed_ms:.0f} ms elapsed)."
                ),
            )
            return []
        except ProviderSearchError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("query-kit search failed: %s", exc)
            _progress(
                "Query-kit search failed",
                (
                    f"{exc}. Continuing without public-provider results "
                    f"after {elapsed_ms:.0f} ms."
                ),
            )
            return []
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "query-kit.search.end query=%r raw_results=%s elapsed_ms=%.0f",
            query,
            len(raw_results),
            elapsed_ms,
        )
        _progress(
            "Query-kit search complete",
            f"{len(raw_results)} raw result(s) in {elapsed_ms:.0f} ms.",
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
            _progress(
                "Query-kit fallback",
                (
                    "Combined provider search failed; retrying individually: "
                    + ",".join(fallback_provider_ids)
                ),
            )

        raw_results: list[Any] = []
        for provider_id in fallback_provider_ids:
            if _remaining_seconds(deadline) <= 0:
                _progress(
                    "Query-kit fallback budget exhausted",
                    (
                        f"Stopped before provider={provider_id}; "
                        "returning partial results."
                    ),
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
                _progress("Query-kit provider failed", f"provider={provider_id}: {exc}")
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
        _progress(
            "Query-kit providers running",
            f"providers={','.join(provider_names)}; budget={remaining:.0f}s",
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
            _progress(
                "Query-kit providers complete",
                f"providers={provider_names[0]}; results={len(results)}",
            )
            return results

        tasks = [
            asyncio.create_task(
                self._search_single_provider(
                    query=query,
                    provider=provider,
                    limit=limit,
                    since_year=since_year,
                    deadline=deadline,
                )
            )
            for provider in provider_instances
        ]
        start = time.perf_counter()
        _done, pending = await asyncio.wait(
            tasks,
            timeout=max(0.1, _remaining_seconds(deadline)),
        )

        provider_result_sets: list[Sequence[Any]] = []
        failures: list[str] = []
        for index, task in enumerate(tasks):
            provider_name = provider_names[index]
            if task in pending:
                task.cancel()
                failures.append(f"{provider_name}: timed out")
                logger.warning("query-kit.provider.timeout provider=%s", provider_name)
                _progress(
                    "Query-kit provider timed out",
                    f"provider={provider_name}; budget exhausted",
                )
                continue
            try:
                provider_result_sets.append(task.result())
            except Exception as exc:
                _status, failure = _provider_failure_outcome(exc)
                failures.append(f"{provider_name}: {failure}")
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
        _progress(
            "Query-kit providers complete",
            (
                f"providers={','.join(provider_names)}; results={len(results)}; "
                f"failures={len(failures)}; elapsed={elapsed_ms:.0f} ms"
            ),
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
        _progress(
            "Query-kit provider running",
            f"provider={provider_name}; budget={remaining:.0f}s",
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
                "query_kit.query": query,
                "query_kit.limit": limit,
                "query_kit.since_year": since_year,
                "query_kit.budget_seconds": remaining,
            },
        ) as provider_span:
            try:
                results = await asyncio.wait_for(
                    self._search_function(
                        query,
                        [provider],
                        limit=limit,
                        since_year=since_year,
                        provider_timeout=remaining,
                    ),
                    timeout=remaining,
                )
            except Exception as exc:
                status, failure = _provider_failure_outcome(exc)
                elapsed_ms = _record_provider_outcome(
                    provider_span,
                    provider=provider_name,
                    query=query,
                    limit=limit,
                    since_year=since_year,
                    start=start,
                    result_count=0,
                    status=status,
                    failure=failure,
                )
                _log_provider_failure(provider_name, elapsed_ms, exc, failure)
                if isinstance(exc, (TimeoutError, ProviderSearchError)):
                    raise
                raise ProviderSearchError(provider_name, failure) from exc
            elapsed_ms = _record_provider_outcome(
                provider_span,
                provider=provider_name,
                query=query,
                limit=limit,
                since_year=since_year,
                start=start,
                result_count=len(results),
                status="ok",
            )
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
        _progress(
            "Query-kit provider complete",
            (
                f"provider={provider_name}; results={len(results)}; "
                f"elapsed={elapsed_ms:.0f} ms"
            ),
        )
        return results

    def _to_search_result(self, result: Any) -> SearchResult:
        title, url, source = str(result.title), str(result.url), str(result.source)
        return SearchResult(
            document_id=_stable_result_id(title=title, url=url, source=source),
            chunk_id="abstract",
            score=1.0,
            title=title,
            source=url,
            snippet=str(getattr(result, "abstract", None) or title),
            metadata={
                "result_type": "query-kit",
                "url": url,
                "provider": source,
                "authors": list(getattr(result, "authors", ()) or ()),
                "year": getattr(result, "year", None),
                "venue": getattr(result, "venue", None),
            },
        )


def _progress(label: str, detail: str) -> None:
    emit_agent_progress(label=label, detail=detail, step_type="retrieval")


def _record_provider_outcome(span: Any, *, start: float, **attrs: Any) -> float:
    elapsed_ms = (time.perf_counter() - start) * 1000
    set_span_attributes(
        span,
        _provider_outcome_attributes(elapsed_ms=elapsed_ms, **attrs),
    )
    return elapsed_ms


def _provider_failure_outcome(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, TimeoutError):
        return "timeout", "timed out"
    if isinstance(exc, ProviderSearchError):
        return "error", str(exc)
    return "error", f"{type(exc).__name__}: {exc}"


def _log_provider_failure(
    provider_name: str,
    elapsed_ms: float,
    exc: Exception,
    failure: str,
) -> None:
    if isinstance(exc, TimeoutError):
        logger.warning(
            "query-kit.provider.timeout provider=%s elapsed_ms=%.0f",
            provider_name,
            elapsed_ms,
        )
        _progress(
            "Query-kit provider timed out",
            f"provider={provider_name} after {elapsed_ms:.0f} ms",
        )
        return
    if isinstance(exc, ProviderSearchError):
        logger.warning(
            "query-kit.provider.failed provider=%s elapsed_ms=%.0f error=%s",
            provider_name,
            elapsed_ms,
            exc,
        )
    else:
        logger.warning(
            "query-kit.provider.failed provider=%s elapsed_ms=%.0f error=%s: %s",
            provider_name,
            elapsed_ms,
            type(exc).__name__,
            exc,
        )
    _progress("Query-kit provider failed", f"provider={provider_name}: {failure}")


def _provider_outcome_attributes(
    *,
    provider: str,
    query: str,
    limit: int,
    since_year: int | None,
    elapsed_ms: float,
    result_count: int,
    status: str,
    failure: str | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "query_kit.provider": provider,
        "query_kit.query": query,
        "query_kit.limit": limit,
        "query_kit.elapsed_ms": round(elapsed_ms, 3),
        "query_kit.result_count": result_count,
        "query_kit.status": status,
    }
    if since_year is not None:
        attrs["query_kit.since_year"] = since_year
    if failure:
        attrs["query_kit.failure"] = failure
        attrs["query_kit.failure_category"] = _failure_category(failure)
        http_status = _http_status_from_message(failure)
        if http_status is not None:
            attrs["query_kit.http_status"] = http_status
    return attrs


def _failure_category(message: str) -> str:
    lowered = message.casefold()
    if "429" in lowered or "rate limit" in lowered or "rate limited" in lowered:
        return "rate_limit"
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if "waf" in lowered or "challenge" in lowered:
        return "waf_challenge"
    if "invalid json" in lowered or "parse" in lowered or "malformed" in lowered:
        return "parse_error"
    if "http 5" in lowered:
        return "server_error"
    if "http 4" in lowered:
        return "client_error"
    return "provider_error"


def _http_status_from_message(message: str) -> int | None:
    import re

    match = re.search(r"HTTP\s+(\d{3})", message, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


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


def _normalize_provider_ids(
    provider_ids: Sequence[str],
    *,
    default_provider_ids: Sequence[str],
) -> list[str]:
    resolved: list[str] = []
    for provider_id in provider_ids:
        normalized = " ".join(str(provider_id).strip().casefold().split())
        normalized = PROVIDER_ALIASES.get(normalized, normalized.replace("_", "-"))
        if normalized == "all":
            return ["all"]
        if normalized not in KNOWN_PROVIDER_IDS:
            logger.warning(
                "query-kit.provider.ignored_unknown provider=%s", provider_id
            )
            continue
        if normalized not in resolved:
            resolved.append(normalized)
    if resolved or list(provider_ids) == list(default_provider_ids):
        return resolved or ["all"]
    return _normalize_provider_ids(
        default_provider_ids,
        default_provider_ids=default_provider_ids,
    )


def _fallback_provider_ids(provider_ids: Sequence[str]) -> list[str]:
    if "all" in provider_ids:
        return list(FALLBACK_PROVIDER_IDS)
    return list(dict.fromkeys(provider_ids))


def _search_result_to_dict(result: SearchResult) -> dict[str, Any]:
    return {**vars(result), "metadata": dict(result.metadata)}
