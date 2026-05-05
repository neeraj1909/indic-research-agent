from __future__ import annotations

import pytest

from indic_research_agent.retrieval import DocumentChunk, SearchService
from indic_research_agent.retrieval.bm25_index import BM25Index

pytestmark = pytest.mark.unit


def test_exact_keyword_match_ranks_above_unrelated_documents() -> None:
    service = SearchService(
        [
            DocumentChunk(
                document_id="research",
                chunk_id="research-1",
                title="Research providers",
                source="test://research",
                text="query-kit searches arxiv pubmed and semantic scholar papers",
            ),
            DocumentChunk(
                document_id="cache",
                chunk_id="cache-1",
                title="Cache",
                source="test://cache",
                text="redis stores cached tool results with expiration policies",
            ),
            DocumentChunk(
                document_id="agent",
                chunk_id="agent-1",
                title="Agent",
                source="test://agent",
                text="langgraph handles tool calling and state transitions",
            ),
        ]
    )

    results = service.search("arxiv pubmed query-kit", top_k=3)

    assert [result.document_id for result in results][0] == "research"
    assert results[0].score > 0
    assert "query-kit searches" in results[0].snippet


def test_empty_query_returns_no_results() -> None:
    service = SearchService(
        [
            DocumentChunk(
                document_id="bm25",
                chunk_id="bm25-1",
                text="BM25 keyword retrieval without embeddings",
            )
        ]
    )

    assert service.search("   ") == []


def test_unindexed_search_returns_no_results() -> None:
    assert BM25Index().search("anything") == []


def test_top_k_must_be_positive() -> None:
    service = SearchService(
        [
            DocumentChunk(
                document_id="bm25",
                chunk_id="bm25-1",
                text="BM25 keyword retrieval without embeddings",
            )
        ]
    )

    with pytest.raises(ValueError, match="top_k"):
        service.search("bm25", top_k=0)
