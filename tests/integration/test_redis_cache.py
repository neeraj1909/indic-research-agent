from __future__ import annotations

import os

import pytest
from redis.asyncio import Redis

from indic_research_agent.services.cache_service import CacheService

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_cache_service_roundtrips_json_with_redis() -> None:
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        pytest.skip("REDIS_URL is required for Redis integration tests")

    redis = Redis.from_url(redis_url, decode_responses=True)
    await redis.flushdb()
    cache = CacheService(redis, default_ttl_seconds=30)
    try:
        payload = {"query": "bm25", "top_k": 5}
        await cache.set_json("tool.search", payload, [{"document_id": "doc-1"}])

        assert await cache.get_json("tool.search", payload) == [
            {"document_id": "doc-1"}
        ]
    finally:
        await cache.close()
