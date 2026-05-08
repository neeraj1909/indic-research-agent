"""LangGraph agent construction."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from indic_research_agent.agent.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    prompt_fingerprint,
)
from indic_research_agent.agent.state import AgentState
from indic_research_agent.services.phoenix_tracing import (
    llm_attributes,
    set_span_output,
    tool_attributes,
    trace_span,
)
from indic_research_agent.tools.fetch import FetchTool
from indic_research_agent.tools.schemas import FetchToolInput, SearchToolInput
from indic_research_agent.tools.search import SearchTool

logger = logging.getLogger(__name__)


def build_agent_graph(
    model: Any,
    *,
    search_tool: SearchTool,
    fetch_tool: FetchTool,
    max_tool_calls: int = 6,
    system_prompt: str = SYSTEM_PROMPT,
):
    """Build the LangGraph tool-calling loop."""

    langchain_tools = _create_langchain_tools(search_tool, fetch_tool)
    system_prompt_version = (
        PROMPT_VERSION if system_prompt == SYSTEM_PROMPT else "custom"
    )
    system_prompt_hash = prompt_fingerprint(system_prompt)
    bound_model = (
        model.bind_tools(langchain_tools) if hasattr(model, "bind_tools") else model
    )

    async def call_model(state: AgentState) -> dict[str, Any]:
        messages = _with_system_prompt(
            state.get("messages", []),
            system_prompt=system_prompt,
        )
        start = time.perf_counter()
        tool_call_count = state.get("tool_call_count", 0)
        logger.info(
            (
                "agent.llm.start messages=%s tool_calls=%s "
                "system_prompt_version=%s system_prompt_hash=%s"
            ),
            len(messages),
            tool_call_count,
            system_prompt_version,
            system_prompt_hash,
        )
        _emit_custom(
            {
                "event": "agent.llm.start",
                "messages": len(messages),
                "tool_call_count": tool_call_count,
                "label": "Final synthesis" if tool_call_count else "LLM call",
            }
        )
        with trace_span(
            "agent.llm",
            kind="LLM",
            input_value=_messages_to_trace_payload(messages),
            attributes={
                **llm_attributes(
                    model_name=_model_name(bound_model),
                    invocation_parameters={
                        "streaming": getattr(bound_model, "streaming", None),
                        "temperature": getattr(bound_model, "temperature", None),
                        "max_tokens": getattr(bound_model, "max_tokens", None),
                    },
                ),
                "agent.system_prompt_version": system_prompt_version,
                "agent.system_prompt_hash": system_prompt_hash,
                "agent.tool_call_count": tool_call_count,
            },
        ) as llm_span:
            response = await _ainvoke(bound_model, messages)
            set_span_output(llm_span, _message_to_trace_payload(response))
        elapsed = time.perf_counter() - start
        requested_tool_calls = len(_tool_calls(response))
        logger.info(
            "agent.llm.end elapsed_seconds=%.2f requested_tool_calls=%s",
            elapsed,
            requested_tool_calls,
        )
        _emit_custom(
            {
                "event": "agent.llm.end",
                "latency_ms": elapsed * 1000,
                "requested_tool_calls": requested_tool_calls,
            }
        )
        update: dict[str, Any] = {"messages": [response]}
        if not _tool_calls(response):
            final_answer = _message_content(response)
            update["final_answer"] = final_answer
            _emit_custom(
                {
                    "event": "agent.answer.finalized",
                    "answer_chars": len(final_answer),
                }
            )
        return update

    async def call_tools(state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        retrieved_context = list(state.get("retrieved_context", []))
        for tool_call in _tool_calls(last_message):
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            start = time.perf_counter()
            logger.info("agent.tool.start name=%s args=%s", tool_name, tool_args)
            _emit_custom(
                {
                    "event": "agent.tool.start",
                    "name": tool_name,
                    "tool_call_id": tool_call.get("id"),
                    "args": tool_args,
                    "args_summary": _summarize_mapping(tool_args),
                }
            )
            with trace_span(
                f"tool.{tool_name}",
                kind="RETRIEVER" if tool_name in {"search", "fetch"} else "TOOL",
                input_value=tool_args,
                attributes=tool_attributes(name=tool_name, parameters=tool_args),
            ) as tool_span:
                if tool_name == "search":
                    result = await search_tool.run(
                        SearchToolInput.model_validate(tool_args)
                    )
                elif tool_name == "fetch":
                    result = await fetch_tool.run(
                        FetchToolInput.model_validate(tool_args)
                    )
                else:
                    raise ValueError(f"unknown tool: {tool_name}")
                set_span_output(
                    tool_span,
                    {
                        "summary": _summarize_tool_result(result),
                        "metadata": _tool_result_metadata(result),
                    },
                )
            elapsed = time.perf_counter() - start
            logger.info(
                "agent.tool.end name=%s elapsed_seconds=%.2f",
                tool_name,
                elapsed,
            )

            payload = _dump_tool_payload(result)
            _emit_custom(
                {
                    "event": "agent.tool.end",
                    "name": tool_name,
                    "tool_call_id": tool_call.get("id"),
                    "args": tool_args,
                    "latency_ms": elapsed * 1000,
                    "result_summary": _summarize_tool_result(result),
                    "result_metadata": _tool_result_metadata(result),
                }
            )
            retrieved_context.append(payload)
            tool_messages.append(
                ToolMessage(
                    content=payload,
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )

        return {
            "messages": tool_messages,
            "tool_call_count": state.get("tool_call_count", 0) + len(tool_messages),
            "retrieved_context": retrieved_context,
        }

    def route_after_model(state: AgentState) -> str:
        if state.get("tool_call_count", 0) >= max_tool_calls:
            return END
        last_message = state["messages"][-1]
        return "tools" if _tool_calls(last_message) else END

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_model)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", route_after_model, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")
    return graph.compile()


def _create_langchain_tools(search_tool: SearchTool, fetch_tool: FetchTool):
    async def search(
        query: str,
        top_k: int = 5,
        source: str = "all",
        providers: list[str] | None = None,
        since_year: int | None = None,
    ) -> list[dict[str, object]]:
        """Search local BM25 chunks and public research providers through query-kit."""

        results = await search_tool.run(
            SearchToolInput(
                query=query,
                top_k=top_k,
                source=source,
                providers=providers,
                since_year=since_year,
            )
        )
        return [result.model_dump() for result in results]

    async def fetch(
        document_id: str,
        chunk_id: str | None = None,
        max_chars: int = 8000,
    ) -> dict[str, object]:
        """Fetch a bounded local document chunk by document and chunk id."""

        result = await fetch_tool.run(
            FetchToolInput(
                document_id=document_id,
                chunk_id=chunk_id,
                max_chars=max_chars,
            )
        )
        return result.model_dump()

    return [
        StructuredTool.from_function(
            coroutine=search,
            name="search",
            description=search.__doc__ or "",
            args_schema=SearchToolInput,
        ),
        StructuredTool.from_function(
            coroutine=fetch,
            name="fetch",
            description=fetch.__doc__ or "",
            args_schema=FetchToolInput,
        ),
    ]


async def _ainvoke(model: Any, messages: list[BaseMessage]) -> BaseMessage:
    if hasattr(model, "ainvoke"):
        return await model.ainvoke(messages)
    return await asyncio.to_thread(model.invoke, messages)


def _with_system_prompt(
    messages: list[BaseMessage],
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[BaseMessage]:
    non_system_messages = [
        message for message in messages if not isinstance(message, SystemMessage)
    ]
    return [SystemMessage(content=system_prompt), *non_system_messages]


def _tool_calls(message: BaseMessage) -> list[dict[str, Any]]:
    if isinstance(message, AIMessage):
        return list(message.tool_calls)
    return list(getattr(message, "tool_calls", []) or [])


def _message_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content)


def _messages_to_trace_payload(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return [_message_to_trace_payload(message) for message in messages]


def _message_to_trace_payload(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": getattr(message, "type", message.__class__.__name__),
        "content": _message_content(message),
    }
    tool_calls = _tool_calls(message)
    if tool_calls:
        payload["tool_calls"] = tool_calls
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
        payload["name"] = message.name
    return payload


def _model_name(model: Any) -> str:
    for attr in ("model", "model_name", "model_id"):
        value = getattr(model, attr, None)
        if value:
            return str(value)
    return model.__class__.__name__


def _dump_tool_payload(result: Any) -> str:
    if isinstance(result, list):
        return json.dumps([item.model_dump() for item in result], sort_keys=True)
    return json.dumps(result.model_dump(), sort_keys=True)


def _emit_custom(payload: dict[str, Any]) -> None:
    try:
        get_stream_writer()(payload)
    except Exception:
        # LangGraph custom writers only exist during stream-mode execution.
        return


def _summarize_mapping(mapping: dict[str, Any]) -> str:
    if not mapping:
        return "{}"
    parts: list[str] = []
    for key, value in mapping.items():
        text = str(value)
        if len(text) > 120:
            text = f"{text[:117]}..."
        parts.append(f"{key}={text}")
    return ", ".join(parts)


def _summarize_tool_result(result: Any) -> str:
    if isinstance(result, list):
        if not result:
            return "0 results"
        titles = [str(getattr(item, "title", "untitled")) for item in result[:3]]
        return f"{len(result)} result(s): " + "; ".join(titles)
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return f"{len(content)} character(s) fetched"
    return "tool completed"


def _tool_result_metadata(result: Any) -> dict[str, Any]:
    if isinstance(result, list):
        return {
            "result_count": len(result),
            "document_ids": [str(getattr(item, "document_id", "")) for item in result],
        }
    metadata: dict[str, Any] = {}
    for key in ("document_id", "chunk_id"):
        value = getattr(result, key, None)
        if value is not None:
            metadata[key] = str(value)
    content = getattr(result, "content", None)
    if isinstance(content, str):
        metadata["content_chars"] = len(content)
    return metadata
