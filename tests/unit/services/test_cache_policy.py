from __future__ import annotations

import pytest

from indic_research_agent.services.cache_policy import CachePolicy

pytestmark = pytest.mark.unit


def test_cache_policy_returns_namespace_ttls() -> None:
    policy = CachePolicy()

    assert policy.ttl_for("query-kit.search") == 900
    assert policy.ttl_for("tool.search") == 300
    assert policy.ttl_for("agent.response") == 300
    assert policy.ttl_for("unknown") == 300


def test_cache_policy_has_no_document_dependent_namespaces() -> None:
    policy = CachePolicy()

    assert policy.document_dependent_namespaces == ()
    assert "tool." + "fetch" not in policy.ttl_by_namespace
