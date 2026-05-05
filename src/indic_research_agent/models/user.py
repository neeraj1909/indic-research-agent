"""User model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from indic_research_agent.db.base import Base

if TYPE_CHECKING:
    from indic_research_agent.models.query import UserQuery


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid(), primary_key=True, default=uuid4)
    external_id: Mapped[str | None] = mapped_column(String(255), unique=True)
    display_name: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    queries: Mapped[list[UserQuery]] = relationship("UserQuery", back_populates="user")
