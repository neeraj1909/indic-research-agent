"""Agent response model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from indic_research_agent.db.base import Base

if TYPE_CHECKING:
    from indic_research_agent.models.query import UserQuery


class AgentResponse(Base):
    __tablename__ = "agent_responses"

    id: Mapped[UUID] = mapped_column(Uuid(), primary_key=True, default=uuid4)
    query_id: Mapped[UUID] = mapped_column(ForeignKey("queries.id"))
    answer: Mapped[str] = mapped_column(Text)
    citations_json: Mapped[list] = mapped_column("citations", JSON, default=list)
    model_metadata_json: Mapped[dict] = mapped_column(
        "model_metadata",
        JSON,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    query: Mapped[UserQuery] = relationship("UserQuery", back_populates="responses")
