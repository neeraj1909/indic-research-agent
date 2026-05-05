from __future__ import annotations

import pytest
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
                '"function":{"arguments":"{\\"query\\":\\"bm25\\","}}]}}]}'
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
            "args": {"query": "bm25", "top_k": 5},
            "id": "call-search",
            "type": "tool_call",
        }
    ]


def test_factory_uses_streaming_proxy_adapter_for_chatgpt_proxy_models() -> None:
    model = create_chat_model(
        AppSettings(
            litellm_model="chatgpt/gpt-5.5",
            litellm_api_key=SecretStr("test-key"),
            litellm_api_base="https://proxy.example.test/v1",
            litellm_streaming=True,
        )
    )

    assert isinstance(model, OpenAICompatibleStreamingChatModel)
