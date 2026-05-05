"""In-memory BM25 index."""

from __future__ import annotations

from collections.abc import Iterable

import bm25s

from indic_research_agent.retrieval.search_service import DocumentChunk, SearchResult
from indic_research_agent.retrieval.tokenizer import tokenize_query, tokenize_texts


class BM25Index:
    """Small adapter around bm25s for app-owned search contracts."""

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._retriever = bm25s.BM25()
        self._is_indexed = False

    def index(self, chunks: Iterable[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        self._retriever = bm25s.BM25()
        self._is_indexed = False
        if not self._chunks:
            return
        corpus = [chunk.text for chunk in self._chunks]
        self._retriever.index(tokenize_texts(corpus), show_progress=False)
        self._is_indexed = True

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not query.strip() or not self._is_indexed:
            return []

        result_ids, scores = self._retriever.retrieve(
            tokenize_query(query),
            k=min(top_k, len(self._chunks)),
            show_progress=False,
        )
        results: list[SearchResult] = []
        for index, score in zip(result_ids[0], scores[0], strict=True):
            chunk = self._chunks[int(index)]
            score_float = float(score)
            if score_float <= 0:
                continue
            results.append(
                SearchResult(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    score=score_float,
                    title=chunk.title,
                    source=chunk.source,
                    snippet=_snippet(chunk.text),
                    metadata=chunk.metadata,
                )
            )
        return results


def _snippet(text: str, *, max_chars: int = 260) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 1].rstrip()}..."
