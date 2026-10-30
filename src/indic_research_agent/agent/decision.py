"""Offline contract for one bounded advisory route decision."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

MAX_REQUEST_CHARS = 12_000
QUESTION_NAME = "route"


class DecisionRoute(StrEnum):
    RESEARCH = "research"
    CLARIFY = "clarify"
    SUPPLIED_TEXT = "supplied_text"
    UNSUPPORTED_LOCAL_DOCUMENTS = "unsupported_local_documents"
    LEGACY = "legacy"


DecisionSource = Literal["controlled", "model"]
DecisionStatus = Literal["ok", "fallback"]


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    text: str
    requested_model: str
    rubric_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or len(self.text) > MAX_REQUEST_CHARS:
            raise ValueError(f"text must be at most {MAX_REQUEST_CHARS} characters")
        if not isinstance(self.requested_model, str) or not self.requested_model:
            raise ValueError("requested_model must be a non-empty string")
        if not isinstance(self.rubric_version, str) or not self.rubric_version:
            raise ValueError("rubric_version must be a non-empty string")


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    route: DecisionRoute
    requested_model: str
    rubric_version: str
    status: DecisionStatus
    source: DecisionSource | None = None
    resolved_model: str | None = None
    fallback_reason: str | None = None


class DecisionProvider(Protocol):
    def __call__(self, request: DecisionRequest) -> Awaitable[Mapping[str, object]]: ...


async def decide(
    request: DecisionRequest,
    provider: DecisionProvider,
    *,
    source: DecisionSource = "controlled",
    timeout_seconds: float = 1.0,
) -> DecisionOutcome:
    """Validate one injected response, falling back conservatively on failure."""
    source = _validated_source(source)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive")
    try:
        async with asyncio.timeout(timeout_seconds):
            response = await provider(request)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        return _fallback(request, "timeout", source)
    except Exception:
        return _fallback(request, "provider_error", source)

    return _validated_outcome(request, response, source)


def _validated_outcome(
    request: DecisionRequest,
    response: Mapping[str, object],
    source: DecisionSource,
) -> DecisionOutcome:
    if not isinstance(response, Mapping):
        return _fallback(request, "invalid_response", source)

    model = response.get("model")
    answers = response.get("answers")
    if model is not None and not isinstance(model, str):
        return _fallback(request, "invalid_response", source)
    if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)):
        return _fallback(request, "invalid_answer", source)
    if len(answers) != 1:
        return _fallback(request, "invalid_answer", source)

    answer = answers[0]
    if not isinstance(answer, Mapping):
        return _fallback(request, "invalid_answer", source)
    if answer.get("type") == "refusal":
        return _fallback(request, "refused", source)
    if (
        answer.get("type") != "choice"
        or answer.get("name") != QUESTION_NAME
        or not isinstance(answer.get("choice"), str)
    ):
        return _fallback(request, "invalid_answer", source)
    try:
        route = DecisionRoute(answer["choice"])
    except (KeyError, ValueError):
        return _fallback(request, "invalid_answer", source)
    if not _valid_probabilities(answer):
        return _fallback(request, "invalid_answer", source)

    return DecisionOutcome(
        route=route,
        requested_model=request.requested_model,
        resolved_model=model,
        status="ok",
        source=source,
        rubric_version=request.rubric_version,
    )


def _valid_probabilities(answer: Mapping[str, object]) -> bool:
    for field in ("confidence",):
        if field in answer and not _finite_probability(answer[field]):
            return False
    probabilities = answer.get("probabilities")
    if probabilities is None:
        return True
    if not isinstance(probabilities, Sequence) or isinstance(
        probabilities, (str, bytes)
    ):
        return False
    labels: set[str] = set()
    allowed = {route.value for route in DecisionRoute}
    for item in probabilities:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("value"), str)
            or item["value"] not in allowed
            or item["value"] in labels
            or "probability" not in item
            or not _finite_probability(item["probability"])
        ):
            return False
        labels.add(item["value"])
    return True


def _finite_probability(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _validated_source(source: object) -> DecisionSource:
    if source == "controlled":
        return "controlled"
    if source == "model":
        return "model"
    raise ValueError("source must be 'controlled' or 'model'")


def _fallback(
    request: DecisionRequest, reason: str, source: DecisionSource
) -> DecisionOutcome:
    return DecisionOutcome(
        route=DecisionRoute.LEGACY,
        requested_model=request.requested_model,
        rubric_version=request.rubric_version,
        status="fallback",
        source=source,
        fallback_reason=reason,
    )
