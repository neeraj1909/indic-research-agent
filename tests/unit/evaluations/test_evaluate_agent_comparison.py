from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from indic_research_agent.agent.decision import DecisionRoute
from indic_research_agent.agent.decisions_live import LiveDecisionResult
from indic_research_agent.config import AppSettings
from indic_research_agent.services.agent_service import AgentService

_SCRIPT = Path(__file__).parents[3] / "scripts" / "evaluate_agent_comparison.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_agent_comparison", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

Frame = _MODULE.Frame
load_frames = _MODULE.load_frames
normalized_em = _MODULE.normalized_em
normalized_reference_contains = _MODULE.normalized_reference_contains
run_comparison = _MODULE.run_comparison
summarize_records = _MODULE.summarize_records
live_decision_call = _MODULE._live_decision_call

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_live_evaluator_uses_shared_litellm_transport(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    async def fake_post(question, model, *, api_key, timeout_seconds):
        assert timeout_seconds == 5.0
        calls.append((question, model, api_key))
        return (
            LiveDecisionResult(
                DecisionRoute.RESEARCH,
                model,
                model,
                {"total_tokens": 2},
            ),
            1.5,
            200,
        )

    monkeypatch.setenv("TEST_DECISIONS_KEY", "explicit-test-key")
    monkeypatch.setattr(
        _MODULE,
        "get_settings",
        lambda: AppSettings(_env_file=None, decisions_api_key=None),
    )
    monkeypatch.setattr(_MODULE, "post_decision", fake_post)
    result = await live_decision_call("gpt-6-luna", "TEST_DECISIONS_KEY")(
        "Find sources"
    )

    assert result == {
        "status": "ok",
        "route": "research",
        "resolved_model": "gpt-6-luna",
        "elapsed_ms": 1.5,
        "usage": {"total_tokens": 2},
    }
    assert calls == [("Find sources", "gpt-6-luna", "explicit-test-key")]


class FakeGraph:
    def __init__(self, answer: str, *, fail: bool = False) -> None:
        self.answer = answer
        self.fail = fail

    async def astream(self, *_args, **_kwargs):
        if self.fail:
            raise RuntimeError("controlled graph failure")
        yield (
            "updates",
            {
                "final_answer": self.answer,
                "retrieved_context": [
                    json.dumps(
                        {"results": [{"document_id": "doc-1", "title": "Source"}]}
                    )
                ],
                "tool_call_count": 0,
            },
        )


def service_factory_with_real_decision_mode(modes: list[str], decisions: list[str]):
    def factory(mode: str) -> AgentService:
        modes.append(mode)
        settings = AppSettings(
            _env_file=None,
            decisions_mode="active" if mode == "with_decisions" else "off",
        )

        async def provider(request):
            decisions.append(request.text)
            return {
                "model": "test",
                "answers": [
                    {
                        "type": "choice",
                        "name": "route",
                        "choice": "research",
                        "probabilities": [],
                    }
                ],
            }

        recorder = _MODULE._RecordingDecisionProvider(
            lambda question: provider(type("Request", (), {"text": question})())
        )
        service = AgentService(
            FakeGraph("answer from " + mode),
            settings=settings,
            decisions_provider=recorder if mode == "with_decisions" else None,
        )
        if mode == "with_decisions":
            service._comparison_decision_recorder = recorder
        return service

    return factory


@pytest.mark.asyncio
async def test_pairs_real_agent_service_both_arms_and_preserves_nonidentical_answer():
    modes: list[str] = []
    decisions: list[str] = []

    records = await run_comparison(
        [Frame(0, {"id": "case-1", "question": "Find sources"})],
        service_factory=service_factory_with_real_decision_mode(modes, decisions),
        model="test-model",
        repetitions=1,
        limit=1,
    )

    assert modes == ["no_decisions", "with_decisions"] or modes == [
        "with_decisions",
        "no_decisions",
    ]
    assert decisions == ["Find sources"]
    assert {record["arm"] for record in records} == {"no_decisions", "with_decisions"}
    assert all(record["status"] == "completed" for record in records)
    assert all(record["citations_present"] for record in records)
    assert all(record["normalized_arm_equivalence_em"] is False for record in records)
    assert all(
        record["decision_request_count"]
        == (1 if record["arm"] == "with_decisions" else 0)
        for record in records
    )
    assert all(record["needs_human_review"] for record in records)


@pytest.mark.asyncio
async def test_failure_and_missing_case_are_retained_without_breaking_pair() -> None:
    def factory(mode: str) -> AgentService:
        settings = AppSettings(
            _env_file=None,
            decisions_mode="active" if mode == "with_decisions" else "off",
        )

        async def decision(_question: str):
            return {"status": "ok", "route": "research", "resolved_model": "test"}

        recorder = _MODULE._RecordingDecisionProvider(decision)
        service = AgentService(
            FakeGraph("ok", fail=mode == "no_decisions"),
            settings=settings,
            decisions_provider=recorder if mode == "with_decisions" else None,
        )
        if mode == "with_decisions":
            service._comparison_decision_recorder = recorder
        return service

    records = await run_comparison(
        [
            Frame(0, {"id": "failed", "question": "Q"}),
            Frame(1, {"id": "missing"}),
        ],
        service_factory=factory,
        model="test-model",
        repetitions=1,
        limit=2,
    )

    by_key = {(record["case_id"], record["arm"]): record for record in records}
    assert by_key["failed", "no_decisions"]["failure_class"] == "agent_failed"
    assert by_key["failed", "with_decisions"]["status"] == "completed"
    assert by_key["missing", "no_decisions"]["failure_class"] == "missing_question"
    assert by_key["missing", "with_decisions"]["failure_class"] == "missing_question"
    assert summarize_records(records)["pairs"] == 2


def test_normalized_em_and_summary_denominators_are_conservative() -> None:
    assert normalized_em("A  B", "a b") is True
    assert normalized_em("A", "B") is False
    assert normalized_em("", "B") is None
    summary = summarize_records(
        [
            {
                "pair_id": "1",
                "arm": "no_decisions",
                "status": "completed",
                "citations_present": False,
                "normalized_arm_equivalence_em": True,
                "normalized_reference_exact_match": True,
                "normalized_reference_contained_in_answer": True,
                "needs_human_review": False,
            },
            {
                "pair_id": "1",
                "arm": "with_decisions",
                "status": "completed",
                "citations_present": True,
                "normalized_arm_equivalence_em": True,
                "normalized_reference_exact_match": False,
                "normalized_reference_contained_in_answer": False,
                "needs_human_review": False,
            },
            {
                "pair_id": "2",
                "arm": "no_decisions",
                "status": "failed",
                "citations_present": False,
                "normalized_arm_equivalence_em": None,
                "needs_human_review": True,
            },
            {
                "pair_id": "2",
                "arm": "with_decisions",
                "status": "completed",
                "citations_present": False,
                "normalized_arm_equivalence_em": None,
                "needs_human_review": True,
            },
        ]
    )
    assert summary["normalized_arm_equivalence_em_pairs"] == 1
    assert summary["normalized_arm_equivalence_em_denominator"] == 1
    assert summary["citation_absent_records"] == 3
    assert summary["needs_human_review_pairs"] == 1


def test_loader_rejects_missing_published_frames_without_inventing_cases(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="published FRAMES subset is missing"):
        load_frames(tmp_path / "frames")


def test_reference_metrics_use_gold_not_the_other_arm() -> None:
    assert (
        normalized_reference_contains("The answer is Jane Ballou.", "Jane Ballou")
        is True
    )
    assert normalized_reference_contains("Jane", "Jane Ballou") is False
    records = [
        {
            "pair_id": "1",
            "arm": "no_decisions",
            "status": "completed",
            "citations_present": True,
            "normalized_arm_equivalence_em": False,
            "normalized_reference_exact_match": True,
            "normalized_reference_contained_in_answer": True,
            "needs_human_review": False,
            "decision_request_count": 0,
        },
        {
            "pair_id": "1",
            "arm": "with_decisions",
            "status": "completed",
            "citations_present": True,
            "normalized_arm_equivalence_em": False,
            "normalized_reference_exact_match": False,
            "normalized_reference_contained_in_answer": True,
            "needs_human_review": False,
            "decision_request_count": 1,
        },
    ]
    summary = summarize_records(records)
    assert summary["by_arm"]["no_decisions"]["reference_exact_match"] == 1
    assert summary["by_arm"]["with_decisions"]["reference_exact_match"] == 0
    assert summary["by_arm"]["with_decisions"]["decision_request_count"] == 1


def test_live_factory_pins_selected_generation_key_and_separate_model(monkeypatch):
    from pydantic import SecretStr

    constructed = []
    configured = AppSettings(_env_file=None)
    monkeypatch.setattr(_MODULE, "get_settings", lambda: configured)
    monkeypatch.setenv("SELECTED_OPENAI_KEY", "selected-test-key")
    monkeypatch.setenv("GENERATION_KEY", "generation-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://inherited.invalid/v1")
    monkeypatch.setattr(_MODULE, "_live_decision_call", lambda *a, **kw: None)

    class CapturingService:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setattr(_MODULE, "AgentService", CapturingService)
    factory = _MODULE._build_service_factory(
        "gpt-6-luna",
        "SELECTED_OPENAI_KEY",
        generation_model="gpt-4.1-mini",
        generation_api_key_env="GENERATION_KEY",
    )
    factory("no_decisions")
    factory("with_decisions")
    for call in constructed:
        settings = call["settings"]
        assert settings.openai_model == "gpt-4.1-mini"
        assert settings.openai_api_key == SecretStr("generation-test-key")
        assert settings.decisions_api_key == SecretStr("selected-test-key")
    assert constructed[0]["settings"].decisions_mode == "off"
    assert constructed[1]["settings"].decisions_mode == "active"


@pytest.mark.asyncio
async def test_generation_count_is_unknown_when_service_has_no_counter():
    service = AgentService(FakeGraph("answer"), settings=AppSettings(_env_file=None))
    result = await _MODULE._collect(service, "Find sources")
    assert result["generation_invocations"] is None
    assert (
        result["generation_invocations_status"] == "unavailable_from_service_contract"
    )
    assert result["observed_graph_llm_starts"] == 0
    assert result["usage"] is None


@pytest.mark.asyncio
async def test_failed_pair_is_checkpointed_before_stopping():
    checkpoints = []
    decisions = []
    modes = []
    underlying = service_factory_with_real_decision_mode(modes, decisions)

    def factory(mode):
        service = underlying(mode)
        service._graph = FakeGraph("answer", fail=True)
        return service

    records = await run_comparison(
        [
            Frame(0, {"id": "first", "question": "Find sources"}),
            Frame(1, {"id": "second", "question": "Find more sources"}),
        ],
        service_factory=factory,
        model="test",
        repetitions=1,
        limit=2,
        checkpoint=lambda rows: checkpoints.append(list(rows)),
        stop_on_failure=True,
    )
    assert len(records) == 2
    assert len(checkpoints) == 1
    assert checkpoints[0] == records
    assert {row["case_id"] for row in records} == {"first"}
