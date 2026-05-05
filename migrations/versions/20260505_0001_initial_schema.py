"""initial schema

Revision ID: 20260505_0001
Revises:
Create Date: 2026-05-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260505_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=True),
        sa.Column("display_name", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("external_id", name=op.f("uq_users_external_id")),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("source_uri", sa.String(length=2000), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint("source_uri", name=op.f("uq_documents_source_uri")),
    )
    op.create_table(
        "cache_metadata",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("namespace", sa.String(length=100), nullable=False),
        sa.Column("cache_key", sa.String(length=500), nullable=False),
        sa.Column("ttl_seconds", sa.Integer(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("hit_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_cache_metadata")),
        sa.UniqueConstraint("namespace", "cache_key", name="uq_cache_metadata_key"),
    )
    op.create_table(
        "queries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("session_id", sa.String(length=255), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("rewritten_text", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_queries_user_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_queries")),
    )
    op.create_index(
        op.f("ix_queries_session_id"), "queries", ["session_id"], unique=False
    )
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_key", sa.String(length=255), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("start_offset", sa.Integer(), nullable=True),
        sa.Column("end_offset", sa.Integer(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_chunks_document_id_documents"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_chunks")),
        sa.UniqueConstraint("document_id", "chunk_key", name="uq_document_chunks_key"),
    )
    op.create_table(
        "agent_responses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("query_id", sa.Uuid(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("model_metadata", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["query_id"],
            ["queries.id"],
            name=op.f("fk_agent_responses_query_id_queries"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_responses")),
    )
    op.create_table(
        "tool_calls",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("query_id", sa.Uuid(), nullable=False),
        sa.Column("tool_name", sa.String(length=100), nullable=False),
        sa.Column("arguments", sa.JSON(), nullable=False),
        sa.Column("result_summary", sa.JSON(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["query_id"], ["queries.id"], name=op.f("fk_tool_calls_query_id_queries")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tool_calls")),
    )


def downgrade() -> None:
    op.drop_table("tool_calls")
    op.drop_table("agent_responses")
    op.drop_table("document_chunks")
    op.drop_index(op.f("ix_queries_session_id"), table_name="queries")
    op.drop_table("queries")
    op.drop_table("cache_metadata")
    op.drop_table("documents")
    op.drop_table("users")
