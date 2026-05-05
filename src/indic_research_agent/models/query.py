"""User query model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from indic_research_agent.db.base import Base

if TYPE_CHECKING:
    from indic_research_agent.models.agent_response import AgentResponse
    from indic_research_agent.models.tool_call import ToolCall
    from indic_research_agent.models.user import User


class UserQuery(Base):
    __tablename__ = "queries"

    id: Mapped[UUID] = mapped_column(Uuid(), primary_key=True, default=uuid4)
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    session_id: Mapped[str | None] = mapped_column(String(255), index=True)
    text: Mapped[str] = mapped_column(Text)
    rewritten_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    user: Mapped[User | None] = relationship("User", back_populates="queries")
    tool_calls: Mapped[list[ToolCall]] = relationship(
        "ToolCall",
        back_populates="query",
    )
    responses: Mapped[list[AgentResponse]] = relationship(
        "AgentResponse",
        back_populates="query",
    )
