from __future__ import annotations

import pytest

from indic_research_agent.services.cache_service import build_cache_key

pytestmark = pytest.mark.unit


def test_cache_key_is_stable_for_equivalent_payloads() -> None:
    left = build_cache_key("tool.search", {"b": 2, "a": {"x": 1}})
    right = build_cache_key("tool.search", {"a": {"x": 1}, "b": 2})

    assert left == right
    assert left.startswith("tool.search:")


def test_cache_key_namespace_changes_key() -> None:
    payload = {"query": "Hindi OCR"}

    assert build_cache_key("one", payload) != build_cache_key("two", payload)


def test_cache_key_has_bounded_length() -> None:
    key = build_cache_key("tool.search", {"query": "Hindi OCR" * 100})

    assert len(key) < 100
