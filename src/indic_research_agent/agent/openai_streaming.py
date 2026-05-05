"""OpenAI-compatible streaming chat client for LiteLLM proxy model IDs."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class OpenAICompatibleStreamingChatModel:
    """Small async model adapter for OpenAI-compatible streaming endpoints.

    Some LiteLLM proxy model IDs are named ``chatgpt/...``. Passing those IDs
    through the LiteLLM Python SDK can collide with LiteLLM's built-in ChatGPT
    provider and trigger an interactive device-login flow. This adapter talks
    to the configured proxy endpoint directly while preserving the LangGraph
    ``bind_tools`` / ``ainvoke`` surface used by the app.
    """

    model: str
    api_key: str
    api_base: str
    temperature: float = 0.2
    request_timeout: float = 30.0
    max_tokens: int | None = 1200
    streaming: bool = True
    tools: Sequence[BaseTool] = ()

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        **_: Any,
    ) -> OpenAICompatibleStreamingChatModel:
        return replace(self, tools=tuple(tools))

    async def ainvoke(self, messages: list[BaseMessage], **_: Any) -> AIMessage:
        return await asyncio.to_thread(self._invoke_sync, messages)

    def _invoke_sync(self, messages: list[BaseMessage]) -> AIMessage:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_openai(message) for message in messages],
            "stream": self.streaming,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.tools:
            payload["tools"] = [_tool_to_openai(tool) for tool in self.tools]

        request = urllib.request.Request(
            _chat_completions_url(self.api_base),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if self.streaming else "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout,
            ) as response:
                if self.streaming:
                    return _ai_message_from_stream(
                        raw.decode("utf-8", "replace") for raw in response
                    )
                return _ai_message_from_payload(
                    json.loads(response.read().decode("utf-8"))
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"OpenAI-compatible chat request failed with HTTP {exc.code}: {body}"
            ) from exc


def _chat_completions_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _message_to_openai(message: BaseMessage) -> dict[str, Any]:
    content = _string_content(message.content)
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": content}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": message.tool_call_id,
        }
    if isinstance(message, AIMessage):
        payload: dict[str, Any] = {"role": "assistant", "content": content}
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call.get("args", {})),
                    },
                }
                for tool_call in message.tool_calls
            ]
        return payload
    return {"role": getattr(message, "type", "user"), "content": content}


def _string_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content)


def _tool_to_openai(tool: BaseTool) -> dict[str, Any]:
    schema = (
        tool.args_schema.model_json_schema()
        if tool.args_schema is not None
        else {"type": "object", "properties": {}}
    )
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }


def _ai_message_from_payload(payload: dict[str, Any]) -> AIMessage:
    message = payload.get("choices", [{}])[0].get("message", {})
    return AIMessage(
        content=message.get("content") or "",
        tool_calls=_tool_calls_from_openai(message.get("tool_calls") or []),
    )


def _ai_message_from_stream(lines: Iterable[str]) -> AIMessage:
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue
        chunk = json.loads(data)
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        if delta.get("content"):
            content_parts.append(delta["content"])
        for tool_call in delta.get("tool_calls") or []:
            index = int(tool_call.get("index", 0))
            existing = tool_calls.setdefault(
                index,
                {"id": None, "name": None, "arguments": ""},
            )
            if tool_call.get("id"):
                existing["id"] = tool_call["id"]
            function = tool_call.get("function") or {}
            if function.get("name"):
                existing["name"] = function["name"]
            if function.get("arguments"):
                existing["arguments"] += function["arguments"]

    return AIMessage(
        content="".join(content_parts),
        tool_calls=[
            {
                "id": value["id"] or f"call_{index}",
                "name": value["name"] or "",
                "args": _json_object_or_empty(value["arguments"]),
            }
            for index, value in sorted(tool_calls.items())
        ],
    )


def _tool_calls_from_openai(
    raw_tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for index, raw_tool_call in enumerate(raw_tool_calls):
        function = raw_tool_call.get("function") or {}
        tool_calls.append(
            {
                "id": raw_tool_call.get("id") or f"call_{index}",
                "name": function.get("name") or "",
                "args": _json_object_or_empty(function.get("arguments") or ""),
            }
        )
    return tool_calls


def _json_object_or_empty(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}
