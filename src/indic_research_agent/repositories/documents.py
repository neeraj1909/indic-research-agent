"""Document repository."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from indic_research_agent.models import Document, DocumentChunk


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_document(
        self,
        *,
        title: str | None,
        source_uri: str | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Document:
        document = Document(
            title=title,
            source_uri=source_uri,
            metadata_json=dict(metadata or {}),
        )
        self._session.add(document)
        await self._session.flush()
        return document

    async def add_chunk(
        self,
        *,
        document_id: UUID,
        chunk_key: str,
        text: str,
        token_count: int | None = None,
        start_offset: int | None = None,
        end_offset: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DocumentChunk:
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_key=chunk_key,
            text=text,
            token_count=token_count,
            start_offset=start_offset,
            end_offset=end_offset,
            metadata_json=dict(metadata or {}),
        )
        self._session.add(chunk)
        await self._session.flush()
        return chunk

    async def list_chunks(self) -> list[DocumentChunk]:
        result = await self._session.execute(
            select(DocumentChunk).options(selectinload(DocumentChunk.document))
        )
        return list(result.scalars())

    async def get_chunk(
        self,
        *,
        document_id: UUID,
        chunk_key: str,
    ) -> DocumentChunk | None:
        result = await self._session.execute(
            select(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.chunk_key == chunk_key,
            )
        )
        return result.scalar_one_or_none()
