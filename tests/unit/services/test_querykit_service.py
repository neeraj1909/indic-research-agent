from __future__ import annotations

import asyncio
from dataclasses import dataclass

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

    async def search_function(query, providers, *, limit, since_year):
        search_calls.append((query, providers, limit, since_year))
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

    assert provider_calls[0][0] == ["arxiv"]
    assert provider_calls[0][1] == pytest.approx(7.0, abs=0.01)
    assert provider_calls[0][2] == {"QUERY_CLI_USER_AGENT": "test"}
    assert search_calls == [("explainable nlp", ["provider"], 3, 2024)]
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

    async def search_function(query, providers, *, limit, since_year):
        if providers == ["all"]:
            raise ProviderSearchError("all", "combined search failed")
        if providers == ["acl"]:
            raise ProviderSearchError("acl", "provider failed")
        if providers == ["arxiv"]:
            return [
                FakeQueryKitResult(
                    title="Fallback paper",
                    url="https://example.test/fallback",
                    source="arxiv",
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

    assert provider_calls[:2] == [["all"], ["arxiv"]]
    assert results[0].title == "Fallback paper"


@pytest.mark.asyncio
async def test_querykit_service_times_out_and_returns_empty_results() -> None:
    def provider_factory(provider_ids, *, timeout, environ):
        return provider_ids

    async def search_function(query, providers, *, limit, since_year):
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
