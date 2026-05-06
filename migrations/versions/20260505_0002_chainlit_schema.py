"""chainlit persistence schema

Revision ID: 20260505_0002
Revises: 20260505_0001
Create Date: 2026-05-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260505_0002"
down_revision = "20260505_0001"
branch_labels = None
depends_on = None

SCHEMA = "chainlit"


def upgrade() -> None:
    op.execute(sa.schema.CreateSchema(SCHEMA, if_not_exists=True))
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("identifier", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("createdAt", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_chainlit_users"),
        sa.UniqueConstraint("identifier", name="uq_chainlit_users_identifier"),
        schema=SCHEMA,
    )
    op.create_table(
        "threads",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("createdAt", sa.Text(), nullable=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("userId", sa.Uuid(), nullable=True),
        sa.Column("userIdentifier", sa.Text(), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(
            ["userId"],
            [f"{SCHEMA}.users.id"],
            name="fk_chainlit_threads_userId_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chainlit_threads"),
        schema=SCHEMA,
    )
    op.create_table(
        "steps",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("threadId", sa.Uuid(), nullable=False),
        sa.Column("parentId", sa.Uuid(), nullable=True),
        sa.Column("streaming", sa.Boolean(), nullable=False),
        sa.Column("waitForAnswer", sa.Boolean(), nullable=True),
        sa.Column("isError", sa.Boolean(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("input", sa.Text(), nullable=True),
        sa.Column("output", sa.Text(), nullable=True),
        sa.Column("createdAt", sa.Text(), nullable=True),
        sa.Column("command", sa.Text(), nullable=True),
        sa.Column("start", sa.Text(), nullable=True),
        sa.Column("end", sa.Text(), nullable=True),
        sa.Column("generation", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("showInput", sa.Text(), nullable=True),
        sa.Column("language", sa.Text(), nullable=True),
        sa.Column("indent", sa.Integer(), nullable=True),
        sa.Column("defaultOpen", sa.Boolean(), nullable=True),
        sa.Column("modes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(
            ["threadId"],
            [f"{SCHEMA}.threads.id"],
            name="fk_chainlit_steps_threadId_threads",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chainlit_steps"),
        schema=SCHEMA,
    )
    op.create_table(
        "elements",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("threadId", sa.Uuid(), nullable=True),
        sa.Column("type", sa.Text(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("chainlitKey", sa.Text(), nullable=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("display", sa.Text(), nullable=True),
        sa.Column("objectKey", sa.Text(), nullable=True),
        sa.Column("size", sa.Text(), nullable=True),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("language", sa.Text(), nullable=True),
        sa.Column("forId", sa.Uuid(), nullable=True),
        sa.Column("mime", sa.Text(), nullable=True),
        sa.Column("props", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("autoPlay", sa.Boolean(), nullable=True),
        sa.Column(
            "playerConfig", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.ForeignKeyConstraint(
            ["threadId"],
            [f"{SCHEMA}.threads.id"],
            name="fk_chainlit_elements_threadId_threads",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chainlit_elements"),
        schema=SCHEMA,
    )
    op.create_table(
        "feedbacks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("forId", sa.Uuid(), nullable=False),
        sa.Column("threadId", sa.Uuid(), nullable=False),
        sa.Column("value", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["threadId"],
            [f"{SCHEMA}.threads.id"],
            name="fk_chainlit_feedbacks_threadId_threads",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chainlit_feedbacks"),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("feedbacks", schema=SCHEMA)
    op.drop_table("elements", schema=SCHEMA)
    op.drop_table("steps", schema=SCHEMA)
    op.drop_table("threads", schema=SCHEMA)
    op.drop_table("users", schema=SCHEMA)
    op.execute(sa.schema.DropSchema(SCHEMA, if_exists=True))
