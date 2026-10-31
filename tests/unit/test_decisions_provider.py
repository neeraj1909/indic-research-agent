from __future__ import annotations

import asyncio
import threading

import pytest

from indic_research_agent.agent import decisions_provider
from indic_research_agent.agent.decision import DecisionRequest, DecisionRoute
from indic_research_agent.agent.decisions_live import LiveDecisionResult
from indic_research_agent.agent.decisions_provider import DecisionsProvider

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_provider_wraps_one_sync_decision_call(monkeypatch) -> None:
    calls = []

    async def fake_post(text, model, *, api_key, timeout_seconds):
        calls.append((text, model, api_key, timeout_seconds))
        return (
            LiveDecisionResult(DecisionRoute.RESEARCH, model, model, {}),
            1.0,
            200,
        )

    monkeypatch.setattr(decisions_provider, "post_decision", fake_post)
    provider = DecisionsProvider(
        api_key="test-key",
        model="gpt-6-luna",
        timeout_seconds=0.5,
        max_concurrency=1,
    )

    response = await provider(DecisionRequest("Find sources", "gpt-6-luna", "route-v1"))

    assert response["model"] == "gpt-6-luna"
    assert response["answers"][0]["choice"] == "research"
    assert calls == [("Find sources", "gpt-6-luna", "test-key", 0.5)]


@pytest.mark.asyncio
async def test_provider_does_not_reveal_missing_key() -> None:
    provider = DecisionsProvider(
        api_key=None,
        model="gpt-6-luna",
        timeout_seconds=0.5,
        max_concurrency=1,
    )

    with pytest.raises(Exception, match="missing_api_key"):
        await provider(DecisionRequest("Find sources", "gpt-6-luna", "route-v1"))


@pytest.mark.asyncio
async def test_cancellation_keeps_slot_until_blocking_call_finishes(
    monkeypatch,
) -> None:
    first_started = threading.Event()
    first_finished = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    calls = 0

    async def fake_post(text, model, *, api_key, timeout_seconds):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            assert release_first.wait(timeout=1)
            first_finished.set()
        else:
            second_started.set()
        return (
            LiveDecisionResult(DecisionRoute.RESEARCH, model, model, {}),
            1.0,
            200,
        )

    monkeypatch.setattr(decisions_provider, "post_decision", fake_post)
    provider = DecisionsProvider(
        api_key="test-key",
        model="gpt-6-luna",
        timeout_seconds=0.5,
        max_concurrency=1,
    )
    request = DecisionRequest("Find sources", "gpt-6-luna", "route-v1")

    first = asyncio.create_task(provider(request))
    await asyncio.to_thread(first_started.wait, 1)
    first.cancel()

    with pytest.raises(asyncio.CancelledError):
        await first

    second = asyncio.create_task(provider(request))
    await asyncio.sleep(0)
    assert not second_started.is_set()
    assert not first_finished.is_set()

    release_first.set()
    await asyncio.wait_for(second, timeout=1)
    assert second_started.is_set()
