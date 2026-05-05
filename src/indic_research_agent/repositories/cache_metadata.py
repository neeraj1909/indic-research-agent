"""Cache metadata repository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from indic_research_agent.models import CacheMetadata


class CacheMetadataRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record_set(
        self,
        *,
        namespace: str,
        cache_key: str,
        ttl_seconds: int | None,
    ) -> CacheMetadata:
        metadata = await self._get(namespace=namespace, cache_key=cache_key)
        expires_at = (
            datetime.now(UTC) + timedelta(seconds=ttl_seconds)
            if ttl_seconds is not None
            else None
        )
        if metadata is None:
            metadata = CacheMetadata(
                namespace=namespace,
                cache_key=cache_key,
                ttl_seconds=ttl_seconds,
                expires_at=expires_at,
            )
            self._session.add(metadata)
        else:
            metadata.ttl_seconds = ttl_seconds
            metadata.expires_at = expires_at
        await self._session.flush()
        return metadata

    async def record_hit(self, *, namespace: str, cache_key: str) -> None:
        metadata = await self._get(namespace=namespace, cache_key=cache_key)
        if metadata is None:
            return
        metadata.hit_count += 1
        await self._session.flush()

    async def _get(
        self,
        *,
        namespace: str,
        cache_key: str,
    ) -> CacheMetadata | None:
        result = await self._session.execute(
            select(CacheMetadata).where(
                CacheMetadata.namespace == namespace,
                CacheMetadata.cache_key == cache_key,
            )
        )
        return result.scalar_one_or_none()
