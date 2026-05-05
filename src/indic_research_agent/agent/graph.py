"""LangGraph agent construction."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph

from indic_research_agent.agent.prompts import SYSTEM_PROMPT
from indic_research_agent.agent.state import AgentState
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
):
    """Build the LangGraph tool-calling loop."""

    langchain_tools = _create_langchain_tools(search_tool, fetch_tool)
    bound_model = (
        model.bind_tools(langchain_tools) if hasattr(model, "bind_tools") else model
    )

    async def call_model(state: AgentState) -> dict[str, Any]:
        messages = _with_system_prompt(state.get("messages", []))
        start = time.perf_counter()
        logger.info(
            "agent.llm.start messages=%s tool_calls=%s",
            len(messages),
            state.get("tool_call_count", 0),
        )
        response = await _ainvoke(bound_model, messages)
        elapsed = time.perf_counter() - start
        logger.info(
            "agent.llm.end elapsed_seconds=%.2f requested_tool_calls=%s",
            elapsed,
            len(_tool_calls(response)),
        )
        update: dict[str, Any] = {"messages": [response]}
        if not _tool_calls(response):
            update["final_answer"] = _message_content(response)
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
            if tool_name == "search":
                result = await search_tool.run(
                    SearchToolInput.model_validate(tool_args)
                )
            elif tool_name == "fetch":
                result = await fetch_tool.run(FetchToolInput.model_validate(tool_args))
            else:
                raise ValueError(f"unknown tool: {tool_name}")
            logger.info(
                "agent.tool.end name=%s elapsed_seconds=%.2f",
                tool_name,
                time.perf_counter() - start,
            )

            payload = _dump_tool_payload(result)
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


def _with_system_prompt(messages: list[BaseMessage]) -> list[BaseMessage]:
    if messages and isinstance(messages[0], SystemMessage):
        return messages
    return [SystemMessage(content=SYSTEM_PROMPT), *messages]


def _tool_calls(message: BaseMessage) -> list[dict[str, Any]]:
    if isinstance(message, AIMessage):
        return list(message.tool_calls)
    return list(getattr(message, "tool_calls", []) or [])


def _message_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content)


def _dump_tool_payload(result: Any) -> str:
    if isinstance(result, list):
        return json.dumps([item.model_dump() for item in result], sort_keys=True)
    return json.dumps(result.model_dump(), sort_keys=True)
