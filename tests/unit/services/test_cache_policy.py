from __future__ import annotations

import pytest

from indic_research_agent.services.cache_policy import CachePolicy
from indic_research_agent.services.document_service import DocumentService

pytestmark = pytest.mark.unit


class FakeCacheService:
    def __init__(self) -> None:
        self.deleted_namespaces: list[str] = []

    async def delete_namespace(self, namespace: str) -> int:
        self.deleted_namespaces.append(namespace)
        return 1


def test_cache_policy_returns_namespace_ttls() -> None:
    policy = CachePolicy()

    assert policy.ttl_for("query-kit.search") == 900
    assert policy.ttl_for("unknown") == 300


@pytest.mark.asyncio
async def test_document_cache_invalidation_deletes_document_dependent_namespaces() -> (
    None
):
    cache = FakeCacheService()
    policy = CachePolicy(document_dependent_namespaces=("tool.search", "tool.fetch"))

    deleted = await DocumentService.invalidate_document_caches(cache, policy)

    assert deleted == 2
    assert cache.deleted_namespaces == ["tool.search", "tool.fetch"]
