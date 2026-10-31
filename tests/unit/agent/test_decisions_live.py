from __future__ import annotations

import asyncio

import httpx
import pytest

from indic_research_agent.agent.decision import DecisionRoute
from indic_research_agent.agent.decisions_live import (
    LiveDecisionError,
    build_payload,
    post_decision,
    validate_response,
)

pytestmark = pytest.mark.unit


def valid_response() -> dict[str, object]:
    return {
        "model": "gpt-6-luna",
        "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
        "answers": [
            {
                "type": "choice",
                "name": "route",
                "choice": "research",
                "confidence": 1.0,
                "probabilities": [
                    {
                        "value": route.value,
                        "probability": 1.0 if route is DecisionRoute.RESEARCH else 0.0,
                    }
                    for route in DecisionRoute
                ],
            }
        ],
    }


def test_payload_is_the_documented_single_choice_envelope() -> None:
    payload = build_payload("Find Hindi OCR research.")
    assert payload["model"] == "gpt-6-luna"
    assert payload["input"] == "Find Hindi OCR research."
    assert len(payload["questions"]) == 1
    assert payload["questions"][0]["name"] == "route"


def test_non_contract_model_is_rejected_before_transport() -> None:
    with pytest.raises(ValueError, match="exactly gpt-6-luna"):
        build_payload("one", "another-model")


@pytest.mark.asyncio
async def test_post_maps_native_litellm_request_and_typed_response() -> None:
    calls: list[dict[str, object]] = []
    typed_response = valid_response()

    async def fake_decisions(**kwargs):
        calls.append(kwargs)
        return typed_response

    result, _latency, status = await post_decision(
        "Find Hindi OCR research.",
        api_key="test-secret",
        decisions_call=fake_decisions,
    )

    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-6-luna"
    assert calls[0]["input"] == "Find Hindi OCR research."
    assert calls[0]["questions"][0]["name"] == "route"
    assert calls[0]["api_key"] == "test-secret"
    assert result.route is DecisionRoute.RESEARCH
    assert result.usage["input_tokens"] == 10
    assert status == 200


@pytest.mark.asyncio
async def test_direct_openai_provider_reaches_decisions_http_boundary(
    monkeypatch,
) -> None:
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(200, json=valid_response())

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "indic_research_agent.agent.decisions_live.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://inherited.invalid/v1")
    result, _, status = await post_decision(
        "Find Hindi OCR research.", api_key="fake-key"
    )
    assert status == 200
    assert result.route is DecisionRoute.RESEARCH
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.openai.com/v1/decisions"
    assert requests[0].headers["Authorization"] == "Bearer fake-key"


@pytest.mark.asyncio
async def test_provider_error_is_sanitized_and_never_retried() -> None:
    calls = 0

    async def fake_decisions(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("secret provider detail")

    with pytest.raises(LiveDecisionError, match="provider_error") as caught:
        await post_decision("one", api_key="secret", decisions_call=fake_decisions)
    assert "secret" not in str(caught.value)
    assert calls == 1


@pytest.mark.asyncio
async def test_timeout_is_sanitized_and_never_retried() -> None:
    calls = 0

    async def fake_decisions(**_kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)

    with pytest.raises(LiveDecisionError, match="timeout"):
        await post_decision(
            "one",
            api_key="secret",
            timeout_seconds=0.001,
            decisions_call=fake_decisions,
        )
    assert calls == 1


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (dict(valid_response(), answers=[]), "malformed_answers"),
        (
            dict(valid_response(), answers=[{"type": "refusal", "name": "route"}]),
            "refusal",
        ),
        (dict(valid_response(), model="other-model"), "model_mismatch"),
        (dict(valid_response(), usage=None), "malformed_usage"),
    ],
)
def test_validation_rejects_non_contract_responses(body, reason) -> None:
    with pytest.raises(LiveDecisionError, match=reason):
        validate_response(body, "gpt-6-luna")


@pytest.mark.parametrize(
    "answer_update",
    [
        {"confidence": True},
        {"probabilities": []},
        {
            "probabilities": [
                {"value": "research", "probability": 1.0},
                {"value": "research", "probability": 0.0},
            ]
        },
        {
            "probabilities": [
                {"value": "research", "probability": 1.0},
                {"value": "not-a-route", "probability": 0.0},
            ]
        },
        {
            "usage": {
                "input_tokens": 1,
                "output_tokens": 0,
                "total_tokens": 1,
                "extra": 2,
            }
        },
        {"usage": {"input_tokens": 1.5, "output_tokens": 0, "total_tokens": 1}},
    ],
)
def test_validation_rejects_non_strict_contract_edges(answer_update) -> None:
    body = valid_response()
    if "usage" in answer_update:
        body["usage"] = answer_update["usage"]
    else:
        body["answers"] = [{**body["answers"][0], **answer_update}]
    with pytest.raises(LiveDecisionError):
        validate_response(body, "gpt-6-luna")


@pytest.mark.asyncio
async def test_missing_key_makes_no_transport_call() -> None:
    calls = 0

    async def fake_decisions(**_kwargs):
        nonlocal calls
        calls += 1

    with pytest.raises(LiveDecisionError, match="missing_api_key"):
        await post_decision("one", api_key="", decisions_call=fake_decisions)
    assert calls == 0
