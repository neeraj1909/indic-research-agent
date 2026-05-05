"""Document service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from indic_research_agent.models import Document, DocumentChunk
from indic_research_agent.repositories import DocumentRepository
from indic_research_agent.retrieval import DocumentChunk as RetrievalDocumentChunk
from indic_research_agent.services.cache_policy import (
    DEFAULT_CACHE_POLICY,
    CachePolicy,
)
from indic_research_agent.services.cache_service import CacheService


@dataclass(frozen=True)
class DocumentChunkInput:
    chunk_key: str
    text: str
    token_count: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    metadata: Mapping[str, Any] | None = None


class DocumentService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        cache_service: CacheService | None = None,
        cache_policy: CachePolicy = DEFAULT_CACHE_POLICY,
    ) -> None:
        self._documents = DocumentRepository(session)
        self._cache_service = cache_service
        self._cache_policy = cache_policy

    async def add_document(
        self,
        *,
        title: str | None,
        source_uri: str | None,
        chunks: Sequence[DocumentChunkInput],
        metadata: Mapping[str, Any] | None = None,
    ) -> Document:
        document = await self._documents.add_document(
            title=title,
            source_uri=source_uri,
            metadata=metadata,
        )
        for chunk in chunks:
            await self._documents.add_chunk(
                document_id=document.id,
                chunk_key=chunk.chunk_key,
                text=chunk.text,
                token_count=chunk.token_count,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                metadata=chunk.metadata,
            )
        if self._cache_service is not None:
            await self.invalidate_document_caches(
                self._cache_service, self._cache_policy
            )
        return document

    async def list_chunks(self) -> list[DocumentChunk]:
        return await self._documents.list_chunks()

    async def list_retrieval_chunks(self) -> list[RetrievalDocumentChunk]:
        chunks = await self.list_chunks()
        return [
            RetrievalDocumentChunk(
                document_id=str(chunk.document_id),
                chunk_id=chunk.chunk_key,
                title=chunk.document.title if chunk.document else None,
                source=chunk.document.source_uri if chunk.document else None,
                text=chunk.text,
                metadata={
                    **(chunk.document.metadata_json if chunk.document else {}),
                    **chunk.metadata_json,
                    "database_chunk_id": str(chunk.id),
                },
            )
            for chunk in chunks
        ]

    @staticmethod
    async def invalidate_document_caches(
        cache_service: CacheService,
        cache_policy: CachePolicy = DEFAULT_CACHE_POLICY,
    ) -> int:
        deleted = 0
        for namespace in cache_policy.document_dependent_namespaces:
            deleted += await cache_service.delete_namespace(namespace)
        return deleted
