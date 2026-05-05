"""LangGraph state contracts."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    query_id: str | None
    tool_call_count: int
    retrieved_context: list[str]
    final_answer: str | None
