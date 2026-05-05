"""Database-backed BM25 index loading."""

from __future__ import annotations

from indic_research_agent.retrieval import SearchService
from indic_research_agent.services.document_service import DocumentService


class DatabaseBM25IndexStore:
    """Build or refresh BM25 search services from persisted document chunks."""

    def __init__(self, document_service: DocumentService) -> None:
        self._document_service = document_service

    async def build_search_service(self) -> SearchService:
        return SearchService(await self._document_service.list_retrieval_chunks())

    async def refresh(self, search_service: SearchService) -> None:
        search_service.refresh(await self._document_service.list_retrieval_chunks())
