"""Cache namespace and TTL policy."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CachePolicy:
    ttl_by_namespace: dict[str, int] = field(
        default_factory=lambda: {
            "query-kit.search": 900,
            "tool.search": 300,
            "tool.fetch": 900,
            "agent.response": 300,
        }
    )
    document_dependent_namespaces: tuple[str, ...] = ("tool.search", "tool.fetch")

    def ttl_for(self, namespace: str) -> int:
        return self.ttl_by_namespace.get(namespace, 300)


DEFAULT_CACHE_POLICY = CachePolicy()
