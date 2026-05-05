"""Tool call audit model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from indic_research_agent.db.base import Base

if TYPE_CHECKING:
    from indic_research_agent.models.query import UserQuery


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[UUID] = mapped_column(Uuid(), primary_key=True, default=uuid4)
    query_id: Mapped[UUID] = mapped_column(ForeignKey("queries.id"))
    tool_name: Mapped[str] = mapped_column(String(100))
    arguments_json: Mapped[dict] = mapped_column("arguments", JSON, default=dict)
    result_summary_json: Mapped[dict] = mapped_column(
        "result_summary",
        JSON,
        default=dict,
    )
    latency_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    query: Mapped[UserQuery] = relationship("UserQuery", back_populates="tool_calls")
