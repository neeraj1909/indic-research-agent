"""Typed tool schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SearchToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    providers: list[str] | None = None
    since_year: int | None = Field(default=None, gt=0)


class ToolSearchResult(BaseModel):
    document_id: str
    chunk_id: str
    score: float
    title: str | None = None
    source: str | None = None
    snippet: str
    citation_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
