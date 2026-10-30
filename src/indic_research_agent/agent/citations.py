"""Run-scoped citation identity for retrieved search results."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from indic_research_agent.tools.schemas import ToolSearchResult

_CITATION_ID = re.compile(r"^S([1-9][0-9]*)$")


def assign_run_citations(
    results: list[ToolSearchResult],
    retrieved_context: list[str],
) -> list[ToolSearchResult]:
    """Copy results and assign stable citation IDs for this run.

    Prior IDs are recovered from serialized result lists in ``retrieved_context``.
    Invalid legacy payloads are deliberately ignored because citation assignment
    must not make a retrieval run fail.
    """
    document_ids: dict[str, str] = {}
    citation_documents: dict[str, str] = {}
    highest_id = 0
    for payload in retrieved_context:
        for item in _context_items(payload):
            document_id = item.get("document_id")
            if not isinstance(document_id, str):
                continue
            citation_ids = _item_citation_ids(item)
            if not citation_ids:
                continue

            citation_id = citation_ids[0]
            previous_id = document_ids.get(document_id)
            previous_document = citation_documents.get(citation_id)
            if previous_id is not None and previous_id != citation_id:
                raise ValueError("contradictory citation identity in retrieved context")
            if previous_document is not None and previous_document != document_id:
                raise ValueError("contradictory citation identity in retrieved context")

            document_ids[document_id] = citation_id
            citation_documents[citation_id] = document_id
            highest_id = max(highest_id, int(citation_id[1:]))

    assigned: list[ToolSearchResult] = []
    for result in results:
        citation_id = document_ids.get(result.document_id)
        if citation_id is None:
            highest_id += 1
            citation_id = f"S{highest_id}"
            document_ids[result.document_id] = citation_id

        copied = result.model_copy(deep=True)
        metadata = dict(copied.metadata)
        metadata["citation_id"] = citation_id
        copied.citation_id = citation_id
        copied.metadata = metadata
        assigned.append(copied)
    return assigned


def _context_items(payload: object) -> Iterable[dict[str, object]]:
    if not isinstance(payload, str):
        return ()
    try:
        decoded = json.loads(payload)
    except (TypeError, ValueError):
        return ()
    if not isinstance(decoded, list):
        return ()
    return (item for item in decoded if isinstance(item, dict))


def _item_citation_ids(item: dict[str, object]) -> tuple[str, ...]:
    candidates: list[object] = [item.get("citation_id")]
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata.get("citation_id"))
    valid_ids = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, str) and _CITATION_ID.fullmatch(candidate)
    )
    if len(set(valid_ids)) > 1:
        raise ValueError("contradictory citation identity in retrieved context")
    return valid_ids[:1]
