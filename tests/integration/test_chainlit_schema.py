from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from indic_research_agent.config import get_settings
from indic_research_agent.db.session import create_engine

pytestmark = pytest.mark.integration


def test_chainlit_schema_migration_creates_isolated_tables(monkeypatch) -> None:
    database_url = os.environ.get("APP_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("APP_DATABASE_URL or DATABASE_URL is required for migration tests")

    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("APP_DATABASE_URL", database_url)
    get_settings.cache_clear()
    try:
        alembic_config = Config(str(repo_root / "alembic.ini"))
        command.upgrade(alembic_config, "head")
        asyncio.run(_assert_chainlit_schema())
    finally:
        get_settings.cache_clear()


async def _assert_chainlit_schema() -> None:
    engine = create_engine(get_settings())
    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                text(
                    """
                    SELECT
                        to_regclass('public.users') AS public_users,
                        to_regclass('public.queries') AS public_queries,
                        to_regclass('public.documents') AS public_documents,
                        to_regclass('public.document_chunks') AS public_document_chunks,
                        to_regclass('chainlit.users') AS chainlit_users,
                        to_regclass('chainlit.threads') AS chainlit_threads,
                        to_regclass('chainlit.steps') AS chainlit_steps,
                        to_regclass('chainlit.elements') AS chainlit_elements,
                        to_regclass('chainlit.feedbacks') AS chainlit_feedbacks,
                        EXISTS (
                            SELECT 1
                            FROM information_schema.columns
                            WHERE table_schema = 'chainlit'
                              AND table_name = 'steps'
                              AND column_name = 'autoCollapse'
                        ) AS has_auto_collapse
                    """
                )
            )
            row = result.mappings().one()
    finally:
        await engine.dispose()

    assert row["public_users"] == "users"
    assert row["public_queries"] == "queries"
    assert row["public_documents"] is None
    assert row["public_document_chunks"] is None
    assert row["chainlit_users"] == "chainlit.users"
    assert row["chainlit_threads"] == "chainlit.threads"
    assert row["chainlit_steps"] == "chainlit.steps"
    assert row["chainlit_elements"] == "chainlit.elements"
    assert row["chainlit_feedbacks"] == "chainlit.feedbacks"
    assert row["has_auto_collapse"] is True
