"""Best-effort LangGraph custom progress event emission.

This helper keeps retrieval/tool code observable during LangGraph streaming while
remaining harmless in unit tests or one-shot execution where no stream writer is
installed.
"""

from __future__ import annotations

from typing import Any

from indic_research_agent.services.agent_events import StepType
from indic_research_agent.services.phoenix_tracing import add_span_event


def emit_agent_progress(
    *,
    label: str,
    detail: str = "",
    step_type: StepType = "run",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit a best-effort custom progress event when LangGraph streaming is active."""

    payload: dict[str, Any] = {
        "event": "agent.progress",
        "label": label,
        "detail": detail,
        "step_type": step_type,
    }
    if metadata:
        payload["metadata"] = metadata
    add_span_event("agent.progress", payload)
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()(payload)
    except Exception:
        # No active LangGraph custom stream writer outside graph.astream(...).
        return
