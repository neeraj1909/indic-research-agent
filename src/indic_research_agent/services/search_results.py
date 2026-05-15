"""Normalized search result contracts shared by research tools/services."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    chunk_id: str
    score: float
    snippet: str
    title: str | None = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
