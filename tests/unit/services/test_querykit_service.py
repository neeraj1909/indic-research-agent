from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from query_cli.domain.errors import ProviderSearchError

from indic_research_agent.config import AppSettings
from indic_research_agent.services.querykit_service import QueryKitService

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class FakeQueryKitResult:
    title: str
    url: str
    source: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None


@pytest.mark.asyncio
async def test_querykit_service_maps_results_without_network() -> None:
    provider_calls = []
    search_calls = []

    def provider_factory(provider_ids, *, timeout, environ):
        provider_calls.append((provider_ids, timeout, environ))
        return ["provider"]

    async def search_function(query, providers, *, limit, since_year, provider_timeout):
        search_calls.append((query, providers, limit, since_year, provider_timeout))
        return [
            FakeQueryKitResult(
                title="Explainable NLP",
                url="https://example.test/paper",
                source="arxiv",
                authors=("A. Researcher",),
                year=2026,
                venue="arXiv",
                abstract="A paper about explainable NLP systems.",
            )
        ]

    service = QueryKitService(
        AppSettings(query_kit_providers="arxiv", query_kit_timeout_seconds=7),
        provider_factory=provider_factory,
        search_function=search_function,
        environ={"QUERY_CLI_USER_AGENT": "test"},
    )

    results = await service.search(
        "explainable nlp",
        providers=["arxiv"],
        limit=3,
        since_year=2024,
    )

    assert provider_calls[0][0] == ["arxiv-web"]
    assert provider_calls[0][1] == pytest.approx(7.0, abs=0.01)
    assert provider_calls[0][2] == {"QUERY_CLI_USER_AGENT": "test"}
    assert search_calls == [
        ("explainable nlp", ["provider"], 3, 2024, pytest.approx(7.0, abs=0.01))
    ]
    assert results[0].document_id.startswith("query-kit:")
    assert results[0].chunk_id == "abstract"
    assert results[0].source == "https://example.test/paper"
    assert results[0].metadata["provider"] == "arxiv"
    assert results[0].metadata["authors"] == ["A. Researcher"]


@pytest.mark.asyncio
async def test_querykit_service_returns_empty_for_blank_query() -> None:
    service = QueryKitService(
        AppSettings(),
        provider_factory=lambda *args, **kwargs: pytest.fail("should not call"),
        search_function=lambda *args, **kwargs: pytest.fail("should not call"),
    )

    assert await service.search("  ") == []


@pytest.mark.asyncio
async def test_querykit_service_retries_all_providers_individually() -> None:
    provider_calls = []

    def provider_factory(provider_ids, *, timeout, environ):
        provider_calls.append(provider_ids)
        return provider_ids

    async def search_function(query, providers, *, limit, since_year, provider_timeout):
        if providers == ["all"]:
            raise ProviderSearchError("all", "combined search failed")
        if providers == ["semantic-scholar"]:
            raise ProviderSearchError("semantic-scholar", "provider failed")
        if providers == ["pubmed"]:
            return [
                FakeQueryKitResult(
                    title="Fallback paper",
                    url="https://example.test/fallback",
                    source="pubmed",
                    abstract="Recovered from a per-provider retry.",
                )
            ]
        return []

    service = QueryKitService(
        AppSettings(query_kit_providers="all"),
        provider_factory=provider_factory,
        search_function=search_function,
    )

    results = await service.search("indic hate speech", limit=2)

    assert provider_calls[:4] == [
        ["all"],
        ["semantic-scholar"],
        ["semantic-scholar-web"],
        ["pubmed"],
    ]
    assert results[0].title == "Fallback paper"


@pytest.mark.asyncio
async def test_querykit_service_times_out_and_returns_empty_results() -> None:
    def provider_factory(provider_ids, *, timeout, environ):
        return provider_ids

    async def search_function(query, providers, *, limit, since_year, provider_timeout):
        await asyncio.sleep(1)
        return [
            FakeQueryKitResult(
                title="Too slow",
                url="https://example.test/slow",
                source="slow",
            )
        ]

    service = QueryKitService(
        AppSettings(query_kit_providers="arxiv", query_kit_timeout_seconds=0.1),
        provider_factory=provider_factory,
        search_function=search_function,
    )

    assert await service.search("indic ocr", limit=1) == []


@pytest.mark.asyncio
async def test_querykit_service_preserves_configured_provider_order() -> None:
    def provider_factory(provider_ids, *, timeout, environ):
        return provider_ids

    async def search_function(query, providers, *, limit, since_year, provider_timeout):
        provider = providers[0]
        if provider == "pubmed":
            await asyncio.sleep(0.02)
        return [
            FakeQueryKitResult(
                title=f"{provider} paper",
                url=f"https://example.test/{provider}",
                source=provider,
                abstract=f"Full abstract from {provider}.",
            )
        ]

    service = QueryKitService(
        AppSettings(
            query_kit_providers="pubmed,arxiv-web",
            query_kit_timeout_seconds=1,
        ),
        provider_factory=provider_factory,
        search_function=search_function,
    )

    results = await service.search("Hindi OCR", limit=2)

    assert [result.metadata["provider"] for result in results] == [
        "pubmed",
        "arxiv-web",
    ]


@pytest.mark.asyncio
async def test_querykit_service_normalizes_llm_supplied_provider_aliases() -> None:
    provider_calls = []

    def provider_factory(provider_ids, *, timeout, environ):
        provider_calls.append(provider_ids)
        return provider_ids

    async def search_function(query, providers, *, limit, since_year, provider_timeout):
        return [
            FakeQueryKitResult(
                title=f"{providers[0]} paper",
                url=f"https://example.test/{providers[0]}",
                source=providers[0],
                abstract="Grounding text.",
            )
        ]

    service = QueryKitService(
        AppSettings(
            query_kit_providers="semantic-scholar,pubmed,arxiv-web",
            query_kit_timeout_seconds=5,
        ),
        provider_factory=provider_factory,
        search_function=search_function,
    )

    await service.search(
        "Indian language hate speech detection",
        providers=["semanticscholar", "crossref", "arxiv"],
        limit=2,
    )

    assert provider_calls[0] == ["semantic-scholar", "arxiv-web"]


@pytest.mark.asyncio
async def test_querykit_service_passes_provider_deadline_to_query_kit() -> None:
    seen_timeouts = []

    def provider_factory(provider_ids, *, timeout, environ):
        return provider_ids

    async def search_function(query, providers, *, limit, since_year, provider_timeout):
        seen_timeouts.append(provider_timeout)
        return [
            FakeQueryKitResult(
                title="Paper",
                url="https://example.test/paper",
                source=providers[0],
                abstract="Grounding text.",
            )
        ]

    service = QueryKitService(
        AppSettings(query_kit_providers="pubmed", query_kit_timeout_seconds=5),
        provider_factory=provider_factory,
        search_function=search_function,
    )

    await service.search("Hindi OCR", limit=1)

    assert seen_timeouts
    assert seen_timeouts[0] == pytest.approx(5.0, abs=0.1)


@pytest.mark.asyncio
async def test_querykit_service_preserves_full_available_abstract_snippet() -> None:
    full_abstract = "Indic OCR grounding text. " * 200

    def provider_factory(provider_ids, *, timeout, environ):
        return provider_ids

    async def search_function(query, providers, *, limit, since_year, provider_timeout):
        return [
            FakeQueryKitResult(
                title="Full abstract paper",
                url="https://example.test/full",
                source="arxiv-web",
                abstract=full_abstract,
            )
        ]

    service = QueryKitService(
        AppSettings(query_kit_providers="arxiv-web", query_kit_timeout_seconds=5),
        provider_factory=provider_factory,
        search_function=search_function,
    )

    results = await service.search("Devanagari OCR", limit=1)

    assert results[0].snippet == full_abstract
    assert len(results[0].snippet) > 1000


@pytest.mark.asyncio
async def test_querykit_service_sets_provider_success_span_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    def capture_attributes(span, attributes):
        captured.append(dict(attributes))

    monkeypatch.setattr(
        "indic_research_agent.services.querykit_service.set_span_attributes",
        capture_attributes,
    )

    def provider_factory(provider_ids, *, timeout, environ):
        return provider_ids

    async def search_function(query, providers, *, limit, since_year, provider_timeout):
        return [
            FakeQueryKitResult(
                title="Paper",
                url="https://example.test/paper",
                source=providers[0],
                abstract="Grounding text.",
            )
        ]

    service = QueryKitService(
        AppSettings(query_kit_providers="pubmed", query_kit_timeout_seconds=5),
        provider_factory=provider_factory,
        search_function=search_function,
    )

    await service.search("Hindi OCR", limit=1, since_year=2020)

    assert any(
        attrs.get("query_kit.status") == "ok"
        and attrs.get("query_kit.provider") == "pubmed"
        and attrs.get("query_kit.query") == "Hindi OCR"
        and attrs.get("query_kit.result_count") == 1
        for attrs in captured
    )


@pytest.mark.asyncio
async def test_querykit_service_sets_provider_failure_span_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    def capture_attributes(span, attributes):
        captured.append(dict(attributes))

    monkeypatch.setattr(
        "indic_research_agent.services.querykit_service.set_span_attributes",
        capture_attributes,
    )

    def provider_factory(provider_ids, *, timeout, environ):
        return provider_ids

    async def search_function(query, providers, *, limit, since_year, provider_timeout):
        raise ProviderSearchError(
            "semantic-scholar",
            "rate limited by upstream provider (HTTP 429)",
            network_failure=True,
        )

    service = QueryKitService(
        AppSettings(
            query_kit_providers="semantic-scholar",
            query_kit_timeout_seconds=5,
        ),
        provider_factory=provider_factory,
        search_function=search_function,
    )

    await service.search("Hindi OCR", limit=1, since_year=2020)

    assert any(
        attrs.get("query_kit.status") == "error"
        and attrs.get("query_kit.provider") == "semantic-scholar"
        and attrs.get("query_kit.failure_category") == "rate_limit"
        and attrs.get("query_kit.http_status") == 429
        and attrs.get("query_kit.result_count") == 0
        for attrs in captured
    )
