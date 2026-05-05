"""Tokenization helpers for lexical search."""

from __future__ import annotations

from collections.abc import Sequence

import bm25s


def tokenize_texts(texts: Sequence[str]):
    """Tokenize corpus texts with deterministic non-verbose defaults."""

    return bm25s.tokenize(
        list(texts),
        lower=True,
        stopwords="english",
        show_progress=False,
        allow_empty=True,
    )


def tokenize_query(query: str):
    """Tokenize a single user query."""

    return bm25s.tokenize(
        query,
        lower=True,
        stopwords="english",
        show_progress=False,
        allow_empty=True,
    )
