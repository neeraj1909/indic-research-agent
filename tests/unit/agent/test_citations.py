from __future__ import annotations

import json

import pytest

from indic_research_agent.agent.citations import assign_run_citations
from indic_research_agent.tools.schemas import ToolSearchResult

pytestmark = pytest.mark.unit


def result(document_id: str, chunk_id: str, **kwargs: object) -> ToolSearchResult:
    return ToolSearchResult(
        document_id=document_id,
        chunk_id=chunk_id,
        score=1.0,
        title=f"Title {document_id}",
        source=f"https://example.test/{document_id}",
        snippet="evidence",
        **kwargs,
    )


def test_assigns_stable_ids_for_chunks_and_across_calls() -> None:
    first = assign_run_citations(
        [result("doc-a", "chunk-1"), result("doc-a", "chunk-2")], []
    )

    second = assign_run_citations(
        [result("doc-b", "chunk-1"), result("doc-a", "chunk-3")],
        [json.dumps([item.model_dump(mode="json") for item in first])],
    )

    assert [item.citation_id for item in first] == ["S1", "S1"]
    assert [item.citation_id for item in second] == ["S2", "S1"]
    assert all(item.metadata["citation_id"] == item.citation_id for item in second)


def test_does_not_mutate_input_or_cached_object() -> None:
    cached = result("doc-a", "chunk-1", citation_id="legacy", metadata={"x": 1})
    before = cached.model_copy(deep=True)

    assigned = assign_run_citations([cached], [])

    assert assigned[0] is not cached
    assert assigned[0].metadata == {"x": 1, "citation_id": "S1"}
    assert cached == before
    assert cached.metadata == {"x": 1}
    assert cached.citation_id == "legacy"


def test_ignores_malformed_legacy_context_safely() -> None:
    assigned = assign_run_citations(
        [result("doc-a", "chunk-1"), result("doc-b", "chunk-1")],
        ["not json", json.dumps({"results": []}), json.dumps([{"bad": True}])],
    )

    assert [item.citation_id for item in assigned] == ["S1", "S2"]


@pytest.mark.parametrize(
    "context",
    [
        [
            {"document_id": "doc-a", "citation_id": "S1"},
            {"document_id": "doc-b", "citation_id": "S1"},
        ],
        [
            {"document_id": "doc-a", "citation_id": "S1"},
            {"document_id": "doc-a", "citation_id": "S2"},
        ],
        [
            {
                "document_id": "doc-a",
                "citation_id": "S1",
                "metadata": {"citation_id": "S2"},
            }
        ],
    ],
)
def test_rejects_ambiguous_prior_identity(context: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="contradictory citation identity"):
        assign_run_citations([result("doc-a", "chunk-1")], [json.dumps(context)])


def test_deep_copy_isolates_nested_metadata() -> None:
    cached = result(
        "doc-a",
        "chunk-1",
        metadata={"nested": {"tags": ["original"]}},
    )

    assigned = assign_run_citations([cached], [])
    assigned[0].metadata["nested"]["tags"].append("changed")  # type: ignore[index]

    assert cached.metadata == {"nested": {"tags": ["original"]}}
