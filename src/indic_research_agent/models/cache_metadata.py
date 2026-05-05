"""Cache metadata model."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

from indic_research_agent.db.base import Base


class CacheMetadata(Base):
    __tablename__ = "cache_metadata"
    __table_args__ = (
        UniqueConstraint("namespace", "cache_key", name="uq_cache_metadata_key"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(), primary_key=True, default=uuid4)
    namespace: Mapped[str] = mapped_column(String(100))
    cache_key: Mapped[str] = mapped_column(String(500))
    ttl_seconds: Mapped[int | None] = mapped_column(Integer)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
