"""add Chainlit step autoCollapse column

Revision ID: 20260505_0003
Revises: 20260505_0002
Create Date: 2026-05-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260505_0003"
down_revision = "20260505_0002"
branch_labels = None
depends_on = None

SCHEMA = "chainlit"


def upgrade() -> None:
    op.add_column(
        "steps",
        sa.Column("autoCollapse", sa.Boolean(), nullable=True),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("steps", "autoCollapse", schema=SCHEMA)
