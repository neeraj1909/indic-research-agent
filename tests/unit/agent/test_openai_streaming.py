from __future__ import annotations

import httpx
import pytest
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

from indic_research_agent.agent.llm import create_chat_model
from indic_research_agent.agent.openai_streaming import (
    OpenAICompatibleStreamingChatModel,
    _ai_message_from_stream,
)
from indic_research_agent.config import AppSettings

pytestmark = pytest.mark.unit


def test_stream_parser_collects_content_chunks() -> None:
    message = _ai_message_from_stream(
        [
            'data: {"choices":[{"delta":{"content":"O"}}]}',
            'data: {"choices":[{"delta":{"content":"K"}}]}',
            "data: [DONE]",
        ]
    )

    assert message.content == "OK"
    assert message.tool_calls == []


def test_stream_parser_collects_tool_call_chunks() -> None:
    message = _ai_message_from_stream(
        [
            (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"id":"call-search","type":"function","function":{"name":"search",'
                '"arguments":""}}]}}]}'
            ),
            (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"{\\"query\\":\\"Hindi OCR\\","}}]}}]}'
            ),
            (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"\\"top_k\\":5}"}}]}}]}'
            ),
            "data: [DONE]",
        ]
    )

    assert message.content == ""
    assert message.tool_calls == [
        {
            "name": "search",
            "args": {"query": "Hindi OCR", "top_k": 5},
            "id": "call-search",
            "type": "tool_call",
        }
    ]


def test_factory_uses_direct_openai_and_ignores_proxy_settings() -> None:
    model = create_chat_model(
        AppSettings(
            _env_file=None,
            openai_api_key=SecretStr("test-key"),
            litellm_api_key=SecretStr("other-key"),
            litellm_api_base="https://proxy.example.test/v1",
        )
    )
    assert isinstance(model, OpenAICompatibleStreamingChatModel)
    assert model.api_base == "https://api.openai.com/v1"
    assert model.api_key == "test-key"
    assert model.model == "gpt-4.1-mini"
    assert "test-key" not in repr(model)


@pytest.mark.asyncio
async def test_direct_generation_http_boundary_is_pinned_and_single_attempt(
    monkeypatch,
) -> None:
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ready"}}]}
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        "indic_research_agent.agent.openai_streaming.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://untrusted.invalid/v1")
    model = create_chat_model(
        AppSettings(
            _env_file=None,
            openai_api_key=SecretStr("test-key"),
            openai_streaming=False,
        )
    )
    answer = await model.ainvoke([HumanMessage(content="hello")])
    assert answer.content == "ready"
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.openai.com/v1/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer test-key"


def test_missing_openai_key_rejects_legacy_proxy_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_chat_model(
            AppSettings(
                _env_file=None, openai_api_key=None, litellm_api_key=SecretStr("proxy")
            )
        )


def test_truncated_stream_is_not_success():
    with pytest.raises(RuntimeError, match="before completion"):
        _ai_message_from_stream(['data: {"choices":[{"delta":{"content":"partial"}}]}'])


@pytest.mark.asyncio
async def test_http_stream_tool_calls_and_redirect_rejection(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"id":"call-1","function":{"name":"search",'
                '"arguments":"{}"}}]}}]}\n\ndata: [DONE]\n\n'
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        "indic_research_agent.agent.openai_streaming.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    model = create_chat_model(
        AppSettings(_env_file=None, openai_api_key=SecretStr("key"))
    )
    answer = await model.ainvoke([HumanMessage(content="find research")])
    assert answer.tool_calls[0]["name"] == "search"
    assert len(requests) == 1

    def redirect(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "https://untrusted.invalid/"})

    monkeypatch.setattr(
        "indic_research_agent.agent.openai_streaming.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(redirect), **kwargs),
    )
    with pytest.raises(RuntimeError, match="request failed"):
        await model.ainvoke([HumanMessage(content="hello")])
    assert len(requests) == 2
