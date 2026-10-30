from __future__ import annotations

import asyncio

import pytest

from indic_research_agent.agent.decision import (
    DecisionRequest,
    DecisionRoute,
    decide,
)

pytestmark = pytest.mark.unit


def request() -> DecisionRequest:
    return DecisionRequest(
        text="Find recent research on Hindi OCR.",
        requested_model="gpt-luna-6.1",
        rubric_version="route-v1",
    )


@pytest.mark.asyncio
async def test_decide_accepts_one_named_route_choice() -> None:
    async def provider(_request: DecisionRequest) -> dict[str, object]:
        return {
            "model": "gpt-6-luna",
            "usage": {"input_tokens": 12, "output_tokens": 2},
            "answers": [
                {
                    "type": "choice",
                    "name": "route",
                    "choice": "research",
                    "confidence": 0.9,
                    "probabilities": [{"value": "research", "probability": 0.9}],
                }
            ],
        }

    outcome = await decide(request(), provider)

    assert outcome.route is DecisionRoute.RESEARCH
    assert outcome.status == "ok"
    assert outcome.source == "controlled"
    assert outcome.requested_model == "gpt-luna-6.1"
    assert outcome.resolved_model == "gpt-6-luna"
    assert outcome.rubric_version == "route-v1"
    assert outcome.fallback_reason is None


@pytest.mark.asyncio
async def test_decide_falls_back_for_malformed_answer() -> None:
    async def provider(_request: DecisionRequest) -> dict[str, object]:
        return {
            "answers": [
                {"type": "choice", "name": "route", "choice": "unknown"},
                {"type": "choice", "name": "route", "choice": "research"},
            ],
        }

    outcome = await decide(request(), provider)

    assert outcome.route is DecisionRoute.LEGACY
    assert outcome.status == "fallback"
    assert outcome.fallback_reason == "invalid_answer"
    assert outcome.requested_model == "gpt-luna-6.1"


@pytest.mark.asyncio
async def test_decide_falls_back_for_timeout_and_propagates_cancellation() -> None:
    async def timeout(_request: DecisionRequest) -> dict[str, object]:
        raise TimeoutError

    outcome = await decide(request(), timeout)
    assert outcome.route is DecisionRoute.LEGACY
    assert outcome.fallback_reason == "timeout"

    async def cancelled(_request: DecisionRequest) -> dict[str, object]:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await decide(request(), cancelled)


@pytest.mark.asyncio
async def test_decide_uses_trusted_source_and_ignores_payload_source() -> None:
    async def provider(_request: DecisionRequest) -> dict[str, object]:
        return {
            "source": "controlled",
            "answers": [{"type": "choice", "name": "route", "choice": "research"}],
        }

    outcome = await decide(request(), provider, source="model")

    assert outcome.status == "ok"
    assert outcome.source == "model"


@pytest.mark.asyncio
async def test_decide_times_out_a_hanging_provider() -> None:
    async def hanging(_request: DecisionRequest) -> dict[str, object]:
        await asyncio.sleep(1)
        return {}

    outcome = await decide(request(), hanging, source="model", timeout_seconds=0.001)

    assert outcome.route is DecisionRoute.LEGACY
    assert outcome.source == "model"
    assert outcome.fallback_reason == "timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["vendor", "", None])
async def test_decide_rejects_invalid_caller_source(source: object) -> None:
    async def provider(_request: DecisionRequest) -> dict[str, object]:
        return {}

    with pytest.raises(ValueError, match="source"):
        await decide(request(), provider, source=source)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_seconds", [0, -1, float("nan"), float("inf")])
async def test_decide_rejects_invalid_timeout(timeout_seconds: float) -> None:
    async def provider(_request: DecisionRequest) -> dict[str, object]:
        return {}

    with pytest.raises(ValueError, match="timeout_seconds"):
        await decide(request(), provider, timeout_seconds=timeout_seconds)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answers", "reason"),
    [
        ([], "invalid_answer"),
        ([{"type": "choice", "name": "other", "choice": "research"}], "invalid_answer"),
        (
            [{"type": "predicate", "name": "route", "choice": "research"}],
            "invalid_answer",
        ),
        ([{"type": "refusal", "name": "route"}], "refused"),
        ([{"type": "choice", "name": "route", "choice": "unknown"}], "invalid_answer"),
        (
            [
                {
                    "type": "choice",
                    "name": "route",
                    "choice": "research",
                    "confidence": float("nan"),
                }
            ],
            "invalid_answer",
        ),
    ],
)
async def test_decide_rejects_invalid_route_answers(
    answers: list[dict[str, object]], reason: str
) -> None:
    async def provider(_request: DecisionRequest) -> dict[str, object]:
        return {"source": "model", "answers": answers}

    outcome = await decide(request(), provider)

    assert outcome.route is DecisionRoute.LEGACY
    assert outcome.status == "fallback"
    assert outcome.fallback_reason == reason


@pytest.mark.asyncio
async def test_decide_reports_refusal_explicitly() -> None:
    async def provider(_request: DecisionRequest) -> dict[str, object]:
        return {"answers": [{"type": "refusal", "refusal": "not allowed"}]}

    outcome = await decide(request(), provider)

    assert outcome.status == "fallback"
    assert outcome.source == "controlled"
    assert outcome.fallback_reason == "refused"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probabilities",
    [
        [{"value": "not-a-route", "probability": 0.5}],
        [
            {"value": "research", "probability": 0.5},
            {"value": "research", "probability": 0.5},
        ],
    ],
)
async def test_decide_rejects_invalid_probability_labels(
    probabilities: list[dict[str, object]],
) -> None:
    async def provider(_request: DecisionRequest) -> dict[str, object]:
        return {
            "answers": [
                {
                    "type": "choice",
                    "name": "route",
                    "choice": "research",
                    "probabilities": probabilities,
                }
            ]
        }

    outcome = await decide(request(), provider)

    assert outcome.route is DecisionRoute.LEGACY
    assert outcome.fallback_reason == "invalid_answer"


def test_decision_request_enforces_explicit_bound() -> None:
    with pytest.raises(ValueError, match="at most 12000"):
        DecisionRequest(
            text="x" * 12_001,
            requested_model="gpt-luna-6.1",
            rubric_version="route-v1",
        )
