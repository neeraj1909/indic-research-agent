"""Chainlit SQLAlchemy data-layer factory."""

from __future__ import annotations

import re
from functools import lru_cache

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

from indic_research_agent.config import AppSettings, get_settings

_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def create_chainlit_data_layer(
    settings: AppSettings | None = None,
) -> SQLAlchemyDataLayer:
    """Create Chainlit's SQLAlchemy data layer in the configured schema."""

    settings = settings or get_settings()
    schema = _validate_schema(settings.chainlit_database_schema)
    return SQLAlchemyDataLayer(
        conninfo=settings.chainlit_database_url,
        connect_args={"server_settings": {"search_path": schema}},
        show_logger=settings.chainlit_data_layer_show_logger,
    )


@lru_cache(maxsize=1)
def get_chainlit_data_layer() -> SQLAlchemyDataLayer:
    """Return one process-local Chainlit data-layer instance."""

    return create_chainlit_data_layer()


def _validate_schema(schema: str) -> str:
    if not _SCHEMA_RE.fullmatch(schema):
        raise ValueError(
            "CHAINLIT_DATABASE_SCHEMA must be a simple PostgreSQL identifier"
        )
    return schema
