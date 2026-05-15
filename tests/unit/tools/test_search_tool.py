from __future__ import annotations

import pytest
from pydantic import ValidationError

from indic_research_agent.services.search_results import SearchResult
from indic_research_agent.tools.schemas import SearchToolInput
from indic_research_agent.tools.search import SearchTool

pytestmark = pytest.mark.unit


class FakeQueryKitService:
    def __init__(self) -> None:
        self.calls = []

    async def search(self, query, *, providers=None, limit=5, since_year=None):
        self.calls.append(query)
        return [
            SearchResult(
                document_id="query-kit:1",
                chunk_id="abstract",
                score=1.0,
                title=f"Research result for {query}",
                source="https://example.test/research",
                snippet="External research abstract.",
                metadata={"providers": providers, "since_year": since_year},
            )
        ][:limit]


class EmptyThenFallbackQueryKitService:
    def __init__(self) -> None:
        self.calls = []

    async def search(self, query, *, providers=None, limit=5, since_year=None):
        self.calls.append(query)
        if query == "Hindi OCR":
            return [
                SearchResult(
                    document_id="query-kit:fallback",
                    chunk_id="abstract",
                    score=1.0,
                    title="Hindi OCR fallback source",
                    source="https://example.test/hindi-ocr",
                    snippet="A full public abstract for a Hindi OCR source.",
                    metadata={"provider": "PubMed"},
                )
            ]
        return []


class FailingQueryKitService:
    async def search(self, query, *, providers=None, limit=5, since_year=None):
        raise TimeoutError("public providers timed out")


def test_search_tool_input_rejects_removed_local_source_mode() -> None:
    assert "source" not in SearchToolInput.model_fields

    with pytest.raises(ValidationError):
        SearchToolInput(query="keyword retrieval", **{"source": "local"})


@pytest.mark.asyncio
async def test_search_tool_returns_querykit_results() -> None:
    querykit_service = FakeQueryKitService()
    tool = SearchTool(querykit_service)

    results = await tool.run(
        SearchToolInput(
            query="xai nlp",
            providers=["arxiv"],
            since_year=2024,
        )
    )

    assert results[0].document_id == "query-kit:1"
    assert results[0].citation_id == "S1"
    assert results[0].metadata["citation_id"] == "S1"
    assert results[0].metadata["providers"] == ["arxiv"]
    assert querykit_service.calls == ["xai nlp"]


@pytest.mark.asyncio
async def test_search_tool_retries_public_search_with_provider_friendly_query() -> None:
    querykit_service = EmptyThenFallbackQueryKitService()
    tool = SearchTool(querykit_service)

    results = await tool.run(
        SearchToolInput(
            query=(
                "Find one recent source on Hindi OCR datasets and answer in two bullets"
            ),
            providers=[
                "semantic-scholar",
                "semantic-scholar-web",
                "pubmed",
                "arxiv-web",
            ],
            since_year=2020,
        )
    )

    assert results[0].document_id == "query-kit:fallback"
    assert querykit_service.calls == [
        "Find one recent source on Hindi OCR datasets and answer in two bullets",
        "Hindi OCR",
    ]


@pytest.mark.asyncio
async def test_search_tool_returns_no_results_when_querykit_fails() -> None:
    tool = SearchTool(FailingQueryKitService())

    results = await tool.run(SearchToolInput(query="Hindi OCR"))

    assert results == []
