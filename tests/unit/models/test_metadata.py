from __future__ import annotations

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

import indic_research_agent.models  # noqa: F401
from indic_research_agent.db.base import Base

pytestmark = pytest.mark.unit


def test_expected_tables_are_registered() -> None:
    assert set(Base.metadata.tables) == {
        "agent_responses",
        "cache_metadata",
        "document_chunks",
        "documents",
        "queries",
        "tool_calls",
        "users",
    }


def test_schema_compiles_for_postgresql() -> None:
    dialect = postgresql.dialect()

    compiled = [
        str(CreateTable(table).compile(dialect=dialect))
        for table in Base.metadata.sorted_tables
    ]

    assert any("CREATE TABLE users" in statement for statement in compiled)
    assert any("CREATE TABLE document_chunks" in statement for statement in compiled)
