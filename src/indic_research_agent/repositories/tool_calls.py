"""Tool call repository."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from indic_research_agent.models import ToolCall


class ToolCallRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(
        self,
        *,
        query_id: UUID,
        tool_name: str,
        arguments: Mapping[str, Any],
        result_summary: Mapping[str, Any],
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> ToolCall:
        tool_call = ToolCall(
            query_id=query_id,
            tool_name=tool_name,
            arguments_json=dict(arguments),
            result_summary_json=dict(result_summary),
            latency_ms=latency_ms,
            error=error,
        )
        self._session.add(tool_call)
        await self._session.flush()
        return tool_call
