from __future__ import annotations

import pytest

from indic_research_agent.retrieval import DocumentChunk, SearchResult, SearchService
from indic_research_agent.tools.schemas import SearchToolInput
from indic_research_agent.tools.search import SearchTool

pytestmark = pytest.mark.unit


class FakeQueryKitService:
    async def search(self, query, *, providers=None, limit=5, since_year=None):
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


@pytest.mark.asyncio
async def test_search_tool_returns_local_bm25_results() -> None:
    tool = SearchTool(
        SearchService(
            [
                DocumentChunk(
                    document_id="bm25",
                    chunk_id="bm25-1",
                    text="BM25 keyword retrieval without vector embeddings",
                    title="BM25",
                )
            ]
        )
    )

    results = await tool.run(SearchToolInput(query="keyword retrieval", source="local"))

    assert results[0].document_id == "bm25"
    assert "keyword retrieval" in results[0].snippet


@pytest.mark.asyncio
async def test_search_tool_can_include_querykit_results() -> None:
    tool = SearchTool(SearchService(), FakeQueryKitService())

    results = await tool.run(
        SearchToolInput(
            query="xai nlp",
            source="research",
            providers=["arxiv"],
            since_year=2024,
        )
    )

    assert results[0].document_id == "query-kit:1"
    assert results[0].metadata["providers"] == ["arxiv"]
