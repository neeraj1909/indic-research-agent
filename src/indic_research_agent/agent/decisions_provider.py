"""Async application adapter for the one-shot OpenAI Decisions transport."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from indic_research_agent.agent.decisions_live import LiveDecisionError, post_decision
from indic_research_agent.config import AppSettings


class DecisionsProvider:
    """Bound one-call adapter with one bounded OpenAI Decisions call."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        timeout_seconds: float,
        max_concurrency: int,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._slots = asyncio.Semaphore(max_concurrency)

    async def __call__(self, request) -> Mapping[str, object]:
        if not self._api_key:
            raise LiveDecisionError("missing_api_key")
        if request.requested_model != self._model:
            raise LiveDecisionError("invalid_model")
        await self._slots.acquire()
        try:
            call_task = asyncio.create_task(
                post_decision(
                    request.text,
                    self._model,
                    api_key=self._api_key,
                    timeout_seconds=self._timeout_seconds,
                )
            )
        except BaseException:
            self._slots.release()
            raise

        def release_slot(completed_task: asyncio.Task) -> None:
            try:
                completed_task.exception()
            except asyncio.CancelledError:
                pass
            finally:
                self._slots.release()

        call_task.add_done_callback(release_slot)
        result, _latency_ms, _status_code = await asyncio.shield(call_task)
        return {
            "model": result.resolved_model,
            "answers": [
                {
                    "type": "choice",
                    "name": "route",
                    "choice": result.route.value,
                }
            ],
            "usage": dict(result.usage) if result.usage is not None else None,
        }


def create_decisions_provider(settings: AppSettings) -> DecisionsProvider:
    """Build the configured provider without exposing its secret in diagnostics."""

    api_key = (
        settings.decisions_api_key.get_secret_value()
        if settings.decisions_api_key is not None
        else None
    )
    return DecisionsProvider(
        api_key=api_key,
        model=settings.decisions_model,
        timeout_seconds=settings.decisions_timeout_seconds,
        max_concurrency=settings.decisions_max_concurrency,
    )
