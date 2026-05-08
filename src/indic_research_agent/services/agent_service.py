"""High-level agent service."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from indic_research_agent.agent.graph import build_agent_graph
from indic_research_agent.agent.llm import create_chat_model
from indic_research_agent.db.session import create_session_factory, session_scope
from indic_research_agent.retrieval import create_seed_search_service
from indic_research_agent.services.agent_events import (
    AgentCompleted,
    AgentFailed,
    AgentProgress,
    AgentRunStarted,
    AgentStreamEvent,
    AgentToken,
    AgentToolFinished,
    AgentToolStarted,
)
from indic_research_agent.services.chat_history import ChatTurn, cap_history
from indic_research_agent.services.phoenix_tracing import (
    force_flush_traces,
    record_span_exception,
    session_attributes,
    set_span_attributes,
    set_span_output,
    trace_span,
)
from indic_research_agent.services.query_service import QueryService
from indic_research_agent.services.querykit_service import QueryKitService
from indic_research_agent.tools.fetch import FetchTool
from indic_research_agent.tools.search import SearchTool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentAnswer:
    answer: str
    retrieved_context: list[str]
    tool_call_count: int


class AgentService:
    """Application-facing wrapper around the LangGraph agent."""

    def __init__(
        self,
        graph=None,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        persist_queries: bool = False,
    ) -> None:
        self._graph = graph if graph is not None else create_default_graph()
        self._session_factory = session_factory
        self._persist_queries = persist_queries

    async def answer(
        self,
        question: str,
        *,
        history: Sequence[ChatTurn] | None = None,
        session_id: str | None = None,
        user_identifier: str | None = None,
    ) -> AgentAnswer:
        """Return the final answer while preserving the legacy one-shot API."""

        completed: AgentCompleted | None = None
        failed: AgentFailed | None = None
        async for event in self.stream_answer(
            question,
            history=history,
            session_id=session_id,
            user_identifier=user_identifier,
        ):
            if isinstance(event, AgentCompleted):
                completed = event
            elif isinstance(event, AgentFailed):
                failed = event
        if failed is not None:
            raise RuntimeError(failed.message)
        if completed is None:
            raise RuntimeError("agent did not produce a final answer")
        return AgentAnswer(
            answer=completed.answer,
            retrieved_context=completed.retrieved_context,
            tool_call_count=completed.tool_call_count,
        )

    async def stream_answer(
        self,
        question: str,
        *,
        history: Sequence[ChatTurn] | None = None,
        session_id: str | None = None,
        user_identifier: str | None = None,
        persist: bool | None = None,
    ) -> AsyncIterator[AgentStreamEvent]:
        """Stream framework-neutral progress, token, and completion events."""

        if not question.strip():
            raise ValueError("question is required")

        history_list = list(history or [])
        should_persist = self._persist_queries if persist is None else persist
        request_id = f"agent-request:{uuid4().hex[:12]}"
        with trace_span(
            "agent.run",
            kind="AGENT",
            input_value={
                "question": question,
                "history": _history_to_trace_payload(history_list),
            },
            attributes={
                **session_attributes(
                    session_id=session_id,
                    user_identifier=user_identifier,
                    metadata={"persist_queries": should_persist},
                    tags=["chainlit", "agent", "bm25-first"],
                ),
                "agent.history_turns": len(history_list),
                "agent.request_id": request_id,
                "agent.session_id": session_id,
            },
        ) as run_span:
            query_id = None
            active_tool_args: dict[str, dict[str, Any]] = {}

            yield AgentRunStarted(
                question=question,
                session_id=session_id,
                user_identifier=user_identifier,
            )
            yield AgentProgress(
                label="Request received",
                detail="Preparing research agent run.",
                step_type="run",
            )

            if should_persist:
                query_id = await self._record_query(
                    text=question,
                    session_id=session_id,
                )
                if query_id is not None:
                    set_span_attributes(
                        run_span,
                        {
                            "agent.query_id": str(query_id),
                            "query.id": str(query_id),
                        },
                    )

            input_state: dict[str, Any] = {
                "messages": _messages_from_history(history_list, question),
                "tool_call_count": 0,
                "retrieved_context": [],
                "final_answer": None,
            }
            collected_state: dict[str, Any] = {
                "messages": [],
                "tool_call_count": 0,
                "retrieved_context": [],
                "final_answer": None,
            }

            try:
                async for raw_chunk in self._stream_graph(input_state, session_id):
                    mode, payload = _split_stream_chunk(raw_chunk)
                    if mode == "custom":
                        event = _event_from_custom_payload(payload)
                        if event is None:
                            continue
                        if isinstance(event, AgentToolStarted):
                            key = event.tool_call_id or event.name
                            active_tool_args[key] = dict(event.arguments)
                        elif isinstance(event, AgentToolFinished):
                            key = event.tool_call_id or event.name
                            if not event.arguments and key in active_tool_args:
                                event = AgentToolFinished(
                                    name=event.name,
                                    latency_ms=event.latency_ms,
                                    result_summary=event.result_summary,
                                    tool_call_id=event.tool_call_id,
                                    arguments=active_tool_args[key],
                                    result_metadata=event.result_metadata,
                                )
                            if should_persist and query_id is not None:
                                await self._record_tool_call(
                                    query_id=query_id,
                                    event=event,
                                )
                        yield event
                    elif mode == "messages":
                        token = _token_from_message_payload(payload)
                        if token:
                            yield AgentToken(text=token)
                    elif mode == "updates":
                        _merge_update_state(collected_state, payload)

                answer = _answer_from_state(collected_state)
                completed = AgentCompleted(
                    answer=answer,
                    retrieved_context=list(
                        collected_state.get("retrieved_context", [])
                    ),
                    tool_call_count=int(collected_state.get("tool_call_count", 0)),
                )
                set_span_output(
                    run_span,
                    {
                        "answer": completed.answer,
                        "retrieved_context_count": len(completed.retrieved_context),
                        "tool_call_count": completed.tool_call_count,
                    },
                )
                if should_persist and query_id is not None:
                    await self._record_response(query_id=query_id, event=completed)
                yield completed
            except Exception as exc:
                logger.exception("agent.stream.failed")
                record_span_exception(run_span, exc)
                set_span_output(run_span, {"error": str(exc)})
                yield AgentFailed(message=str(exc))
            finally:
                force_flush_traces()

    async def _stream_graph(
        self,
        input_state: dict[str, Any],
        session_id: str | None,
    ) -> AsyncIterator[Any]:
        if hasattr(self._graph, "astream"):
            graph_config = (
                {"configurable": {"session_id": session_id}} if session_id else None
            )
            async for chunk in self._graph.astream(
                input_state,
                stream_mode=["updates", "messages", "custom"],
                version="v2",
                config=graph_config,
            ):
                yield chunk
            return

        state = await self._graph.ainvoke(input_state)
        yield ("updates", state)

    async def _record_query(self, *, text: str, session_id: str | None):
        try:
            async with session_scope(self._get_session_factory()) as session:
                query = await QueryService(session).record_query(
                    text=text,
                    session_id=session_id,
                )
                return query.id
        except Exception:
            logger.exception("agent.persistence.query_failed")
            return None

    async def _record_tool_call(self, *, query_id, event: AgentToolFinished) -> None:
        try:
            async with session_scope(self._get_session_factory()) as session:
                await QueryService(session).record_tool_call(
                    query_id=query_id,
                    tool_name=event.name,
                    arguments=event.arguments,
                    result_summary={
                        "summary": event.result_summary,
                        **event.result_metadata,
                    },
                    latency_ms=event.latency_ms,
                )
        except Exception:
            logger.exception("agent.persistence.tool_call_failed")

    async def _record_response(self, *, query_id, event: AgentCompleted) -> None:
        try:
            async with session_scope(self._get_session_factory()) as session:
                await QueryService(session).record_response(
                    query_id=query_id,
                    answer=event.answer,
                    citations=[
                        {"context": context} for context in event.retrieved_context
                    ],
                    model_metadata={"tool_call_count": event.tool_call_count},
                )
        except Exception:
            logger.exception("agent.persistence.response_failed")

    def _get_session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            self._session_factory = create_session_factory()
        return self._session_factory


def create_default_graph():
    search_service = create_seed_search_service()
    search_tool = SearchTool(search_service, QueryKitService())
    fetch_tool = FetchTool(search_service)
    return build_agent_graph(
        create_chat_model(),
        search_tool=search_tool,
        fetch_tool=fetch_tool,
    )


def _messages_from_history(
    history: Sequence[ChatTurn] | None,
    question: str,
) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for turn in cap_history(list(history or [])):
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))
    messages.append(HumanMessage(content=question))
    return messages


def _history_to_trace_payload(history: Sequence[ChatTurn]) -> list[dict[str, str]]:
    return [
        {"role": turn.role, "content": turn.content}
        for turn in cap_history(list(history))
    ]


def _split_stream_chunk(raw_chunk: Any) -> tuple[str, Any]:
    if isinstance(raw_chunk, Mapping):
        chunk_type = raw_chunk.get("type")
        if chunk_type in {"updates", "messages", "custom"}:
            return str(chunk_type), raw_chunk.get("data")
        if str(raw_chunk.get("event", "")).startswith("agent."):
            return "custom", raw_chunk
    if isinstance(raw_chunk, tuple):
        if len(raw_chunk) >= 2 and raw_chunk[-2] in {"updates", "messages", "custom"}:
            return str(raw_chunk[-2]), raw_chunk[-1]
        if len(raw_chunk) == 2 and raw_chunk[0] in {"updates", "messages", "custom"}:
            return str(raw_chunk[0]), raw_chunk[1]
    return "updates", raw_chunk


def _event_from_custom_payload(payload: Any) -> AgentStreamEvent | None:
    if not isinstance(payload, Mapping):
        return None
    event_name = str(payload.get("event", ""))
    if event_name == "agent.llm.start":
        return AgentProgress(
            label=str(payload.get("label") or "LLM call"),
            detail=f"Messages: {payload.get('messages', 0)}",
            step_type="llm",
        )
    if event_name == "agent.llm.end":
        latency_ms = _float_or_none(payload.get("latency_ms"))
        detail = "LLM call complete"
        if latency_ms is not None:
            detail = f"LLM call complete in {latency_ms:.0f} ms"
        return AgentProgress(label="LLM complete", detail=detail, step_type="llm")
    if event_name == "agent.tool.start":
        name = str(payload.get("name") or "tool")
        args = _dict_or_empty(payload.get("args"))
        return AgentToolStarted(
            name=name,
            tool_call_id=_str_or_none(payload.get("tool_call_id")),
            args_summary=str(payload.get("args_summary") or args),
            arguments=args,
        )
    if event_name == "agent.tool.end":
        return AgentToolFinished(
            name=str(payload.get("name") or "tool"),
            tool_call_id=_str_or_none(payload.get("tool_call_id")),
            latency_ms=_float_or_none(payload.get("latency_ms")),
            result_summary=str(payload.get("result_summary") or "tool completed"),
            arguments=_dict_or_empty(payload.get("args")),
            result_metadata=_dict_or_empty(payload.get("result_metadata")),
        )
    if event_name == "agent.answer.finalized":
        return AgentProgress(
            label="Answer finalized",
            detail=f"{payload.get('answer_chars', 0)} characters ready.",
            step_type="run",
        )
    if event_name == "agent.progress":
        return AgentProgress(
            label=str(payload.get("label") or "Agent progress"),
            detail=str(payload.get("detail") or ""),
            step_type=str(payload.get("step_type") or "run"),  # type: ignore[arg-type]
        )
    return AgentProgress(label=event_name or "Agent progress", detail=str(payload))


def _token_from_message_payload(payload: Any) -> str:
    message = payload[0] if isinstance(payload, tuple) and payload else payload
    if not isinstance(message, AIMessage | AIMessageChunk):
        return ""
    if getattr(message, "tool_calls", None):
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _merge_update_state(state: dict[str, Any], update: Any) -> None:
    if not isinstance(update, Mapping):
        return
    candidates: list[Mapping[str, Any]] = []
    if any(
        key in update
        for key in {"messages", "tool_call_count", "retrieved_context", "final_answer"}
    ):
        candidates.append(update)
    for value in update.values():
        if isinstance(value, Mapping):
            candidates.append(value)
    for candidate in candidates:
        if "messages" in candidate:
            messages = candidate.get("messages") or []
            if isinstance(messages, list):
                state.setdefault("messages", [])
                state["messages"].extend(messages)
        if "tool_call_count" in candidate:
            state["tool_call_count"] = candidate.get("tool_call_count") or 0
        if "retrieved_context" in candidate:
            state["retrieved_context"] = list(candidate.get("retrieved_context") or [])
        if candidate.get("final_answer") is not None:
            state["final_answer"] = candidate.get("final_answer")


def _answer_from_state(state: Mapping[str, Any]) -> str:
    if state.get("final_answer"):
        return str(state["final_answer"])
    messages = list(state.get("messages") or [])
    if messages:
        content = getattr(messages[-1], "content", "")
        return content if isinstance(content, str) else str(content)
    return ""


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
