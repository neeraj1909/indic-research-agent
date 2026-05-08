"""Typed tool schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchToolInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    source: Literal["local", "research", "all"] = "all"
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


class FetchToolInput(BaseModel):
    document_id: str = Field(min_length=1)
    chunk_id: str | None = None
    max_chars: int = Field(default=8000, ge=1, le=20000)


class FetchResult(BaseModel):
    document_id: str
    chunk_id: str
    title: str | None = None
    source: str | None = None
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)
