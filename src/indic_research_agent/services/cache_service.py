"""Redis-backed JSON cache service."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from redis.asyncio import Redis

from indic_research_agent.config import AppSettings, get_settings
from indic_research_agent.repositories.cache_metadata import CacheMetadataRepository


class CacheService:
    def __init__(
        self,
        redis: Redis,
        *,
        default_ttl_seconds: int = 300,
        metadata_repository: CacheMetadataRepository | None = None,
    ) -> None:
        self._redis = redis
        self._default_ttl_seconds = default_ttl_seconds
        self._metadata_repository = metadata_repository

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> CacheService:
        settings = settings or get_settings()
        return cls(Redis.from_url(settings.redis_url, decode_responses=True))

    async def get_json(
        self,
        namespace: str,
        payload: Mapping[str, Any],
    ) -> Any | None:
        cache_key = build_cache_key(namespace, payload)
        raw = await self._redis.get(cache_key)
        if raw is None:
            return None
        if self._metadata_repository is not None:
            await self._metadata_repository.record_hit(
                namespace=namespace,
                cache_key=cache_key,
            )
        return json.loads(raw)

    async def set_json(
        self,
        namespace: str,
        payload: Mapping[str, Any],
        value: Any,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        cache_key = build_cache_key(namespace, payload)
        ttl = ttl_seconds or self._default_ttl_seconds
        await self._redis.set(
            cache_key,
            json.dumps(value, sort_keys=True, separators=(",", ":")),
            ex=ttl,
        )
        if self._metadata_repository is not None:
            await self._metadata_repository.record_set(
                namespace=namespace,
                cache_key=cache_key,
                ttl_seconds=ttl,
            )

    async def delete_namespace(self, namespace: str) -> int:
        pattern = f"{namespace}:*"
        deleted = 0
        async for key in self._redis.scan_iter(match=pattern):
            deleted += await self._redis.delete(key)
        return deleted

    async def close(self) -> None:
        await self._redis.aclose()


def build_cache_key(namespace: str, payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return f"{namespace}:{digest}"
