"""Search service data contracts and orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    document_id: str
    chunk_id: str
    text: str
    title: str | None = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id is required")
        if not self.chunk_id.strip():
            raise ValueError("chunk_id is required")
        if not self.text.strip():
            raise ValueError("text is required")


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    chunk_id: str
    score: float
    snippet: str
    title: str | None = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SearchService:
    """Facade over the current keyword retrieval implementation."""

    def __init__(self, chunks: Iterable[DocumentChunk] = ()) -> None:
        from indic_research_agent.retrieval.bm25_index import BM25Index

        self._index = BM25Index()
        self._chunks: list[DocumentChunk] = []
        self._chunks_by_key: dict[tuple[str, str], DocumentChunk] = {}
        self.refresh(chunks)

    def refresh(self, chunks: Iterable[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        self._chunks_by_key = {
            (chunk.document_id, chunk.chunk_id): chunk for chunk in self._chunks
        }
        self._index.index(self._chunks)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        return self._index.search(query, top_k=top_k)

    def fetch(self, document_id: str, chunk_id: str | None = None) -> DocumentChunk:
        matches = [
            chunk
            for chunk in self._chunks
            if chunk.document_id == document_id
            and (chunk_id is None or chunk.chunk_id == chunk_id)
        ]
        if not matches:
            suffix = f" chunk_id={chunk_id!r}" if chunk_id else ""
            raise ValueError(f"document_id={document_id!r}{suffix} was not found")
        return matches[0]


def seed_chunks() -> list[DocumentChunk]:
    """Small local corpus used before database-backed ingestion exists."""

    return [
        DocumentChunk(
            document_id="seed-query-kit",
            chunk_id="seed-query-kit-1",
            title="query-kit research search",
            source="seed://query-kit",
            text=(
                "query-kit searches public research sources including arxiv, "
                "pubmed, semantic scholar, openreview, and ACL Anthology."
            ),
            metadata={"kind": "seed"},
        ),
        DocumentChunk(
            document_id="seed-bm25",
            chunk_id="seed-bm25-1",
            title="BM25 lexical retrieval",
            source="seed://bm25",
            text=(
                "BM25 keyword retrieval ranks document chunks without embedding "
                "models or vector databases."
            ),
            metadata={"kind": "seed"},
        ),
        DocumentChunk(
            document_id="seed-agent",
            chunk_id="seed-agent-1",
            title="LangGraph tool agent",
            source="seed://agent",
            text=(
                "LangGraph coordinates tool calling so an agent can search, "
                "fetch evidence, and answer with citations."
            ),
            metadata={"kind": "seed"},
        ),
    ]


def create_seed_search_service() -> SearchService:
    return SearchService(seed_chunks())
