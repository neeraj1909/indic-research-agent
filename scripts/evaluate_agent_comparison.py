"""Paired OFF versus Decisions-active research-agent evaluation.

This runner is intentionally live-only for generation.  Without ``--live`` it
only validates the published FRAMES input and writes a reproducible manifest;
it never calls OpenAI, LiteLLM, query-kit, or the application service.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
import unicodedata
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from indic_research_agent.agent.decisions_live import (
    DEFAULT_MODEL,
    LiveDecisionError,
    post_decision,
)
from indic_research_agent.config import AppSettings, get_settings
from indic_research_agent.services.agent_events import (
    AgentCompleted,
    AgentFailed,
    AgentProgress,
    AgentToolFinished,
)
from indic_research_agent.services.agent_service import AgentService

FRAMES_PATH = Path("data/evaluations/frames")
MAX_CASES = 50
MAX_REPETITIONS = 5
ARMS = ("no_decisions", "with_decisions")
DecisionCall = Callable[[str], Awaitable[Mapping[str, Any]]]
ServiceFactory = Callable[[str], AgentService]


@dataclass(frozen=True)
class Frame:
    index: int
    raw: Mapping[str, Any]

    @property
    def case_id(self) -> str:
        return str(
            self.raw.get("case_id") or self.raw.get("id") or f"row-{self.index + 1}"
        )

    @property
    def question(self) -> str | None:
        for key in ("question", "request", "prompt", "input"):
            value = self.raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    @property
    def reference_answer(self) -> str | None:
        value = self.raw.get("reference_answer")
        return value if isinstance(value, str) and value.strip() else None


def load_frames(path: Path = FRAMES_PATH) -> list[Frame]:
    """Load published JSON/JSONL FRAMES without reading environment files."""
    if not path.exists():
        raise FileNotFoundError(
            f"published FRAMES subset is missing: {path}; no cases were invented"
        )
    files = (
        [path]
        if path.is_file()
        else sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json"))
    )
    rows: list[Mapping[str, Any]] = []
    for file in files:
        if file.suffix == ".jsonl":
            values = [
                json.loads(line)
                for line in file.read_text().splitlines()
                if line.strip()
            ]
        else:
            parsed = json.loads(file.read_text())
            values = (
                parsed
                if isinstance(parsed, list)
                else parsed.get("cases", parsed.get("frames", []))
            )
        rows.extend(value for value in values if isinstance(value, Mapping))
    return [Frame(index, row) for index, row in enumerate(rows)]


def normalized_answer(value: str | None) -> str | None:
    """Conservative lower-bound normalization: NFKC, casefold, whitespace only."""
    if not isinstance(value, str) or not value.strip():
        return None
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def normalized_em(left: str | None, right: str | None) -> bool | None:
    left_normalized = normalized_answer(left)
    right_normalized = normalized_answer(right)
    if left_normalized is None or right_normalized is None:
        return None
    return left_normalized == right_normalized


def normalized_reference_contains(
    answer: str | None, reference: str | None
) -> bool | None:
    """Return a bounded, conservative reference-contained-in-answer proxy."""
    answer_normalized = normalized_answer(answer)
    reference_normalized = normalized_answer(reference)
    if answer_normalized is None or reference_normalized is None:
        return None
    # Avoid treating short/common fragments (for example "a" or "in") as a hit.
    if len(reference_normalized) < 4:
        return False
    return bool(
        re.search(rf"(?<!\w){re.escape(reference_normalized)}(?!\w)", answer_normalized)
    )


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _citation_records(context: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for payload in context:
        try:
            parsed = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            continue
        items = (
            parsed
            if isinstance(parsed, list)
            else parsed.get("results", [])
            if isinstance(parsed, Mapping)
            else []
        )
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, Mapping):
                continue
            record = {
                key: item[key]
                for key in (
                    "citation_id",
                    "document_id",
                    "title",
                    "url",
                    "source",
                    "provider",
                )
                if key in item and isinstance(item[key], (str, int, float))
            }
            if record:
                records.append(record)
    return records


def _numeric_usage(value: Mapping[str, Any] | None) -> dict[str, int | float] | None:
    if not isinstance(value, Mapping):
        return None
    usage = {
        str(key): item
        for key, item in value.items()
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    }
    return usage or None


def _manifest(
    frames: Sequence[Frame],
    *,
    model: str,
    repetitions: int,
    limit: int,
    generation_model: str | None = None,
) -> dict[str, Any]:
    payload = [frame.raw for frame in frames[:limit]]
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "requested_model": model,
        "generation_model": generation_model,
        "generation_endpoint": "https://api.openai.com/v1",
        "prompt_revision": "application-default",
        "retrieval": {
            "mode": "application-configured-query-kit",
            "frozen_evidence": False,
            "warning": "query-kit results may vary in live comparison",
        },
        "tool_budget": "application-default",
        "repetitions": repetitions,
        "limit": limit,
        "case_count": len(payload),
        "frames_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "paid_work": "live inference and retrieval may incur provider charges",
    }


async def _collect(service: AgentService, question: str) -> dict[str, Any]:
    started = time.perf_counter()
    answer: str | None = None
    context: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    failure: str | None = None
    decision_progress: dict[str, str] | None = None
    observed_llm_starts = 0
    try:
        async for event in service.stream_answer(question, persist=False):
            if isinstance(event, AgentProgress) and event.label == "Decision":
                decision_progress = {
                    key: value
                    for key, value in (
                        part.split("=", 1)
                        for part in event.detail.split()
                        if "=" in part
                    )
                }
            elif (
                isinstance(event, AgentProgress)
                and event.step_type == "llm"
                and (event.label in {"LLM call", "Final synthesis"})
            ):
                observed_llm_starts += 1
            elif isinstance(event, AgentCompleted):
                answer = event.answer
                context = list(event.retrieved_context)
            elif isinstance(event, AgentFailed):
                failure = "agent_failed"
            elif isinstance(event, AgentToolFinished):
                tool_calls.append(
                    {
                        "name": event.name,
                        "tool_call_id": event.tool_call_id,
                        "latency_ms": event.latency_ms,
                        "result_metadata": dict(event.result_metadata),
                    }
                )
    except Exception:
        failure = "agent_exception"
    citations = _citation_records(context)
    return {
        "route": "application_graph",
        "status": "failed" if failure else "completed",
        "failure_class": failure,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "answer": answer,
        "citations": citations,
        "citations_present": bool(citations),
        "tool_calls": tool_calls,
        "usage": None,
        "cost_status": "usage_unavailable",
        "generation_invocations": None,
        "generation_invocations_status": "unavailable_from_service_contract",
        "observed_graph_llm_starts": observed_llm_starts,
        "billing_evidence": "unknown",
        "decision_progress": decision_progress,
    }


async def run_comparison(
    frames: Sequence[Frame],
    *,
    service_factory: ServiceFactory,
    decision_call: DecisionCall | None = None,
    model: str,
    repetitions: int,
    limit: int,
    checkpoint: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
    stop_on_failure: bool = False,
    generation_model: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for frame in frames[:limit]:
        for repetition in range(repetitions):
            pair_id = f"{frame.case_id}:{repetition + 1}"
            order = (
                ARMS
                if int(hashlib.sha256(pair_id.encode()).hexdigest(), 16) % 2 == 0
                else ARMS[::-1]
            )
            arms: dict[str, dict[str, Any]] = {}
            for arm in order:
                base = {
                    "pair_id": pair_id,
                    "case_id": frame.case_id,
                    "arm": arm,
                    "repetition": repetition + 1,
                    "requested_model": model,
                    "generation_model": generation_model,
                    "resolved_model": None,
                    "decision_status": "not_run",
                    "decision_route": None,
                    "decision_usage": None,
                    "reference_answer": frame.reference_answer,
                }
                if frame.question is None:
                    base.update(status="invalid_case", failure_class="missing_question")
                    arms[arm] = base
                    continue
                service = service_factory(arm)
                recorder = getattr(service, "_comparison_decision_recorder", None)
                if arm == "with_decisions" and recorder is None:
                    base.update(
                        status="failed", failure_class="active_provider_missing"
                    )
                    arms[arm] = base
                    continue
                result = await _collect(service, frame.question)
                base.update(result)
                if recorder is not None and recorder.last is not None:
                    base.update(
                        route=recorder.last.get("route"),
                        decision_status=recorder.last.get("status", "ok"),
                        decision_route=recorder.last.get("route"),
                        resolved_model=recorder.last.get("resolved_model"),
                        decision_usage=_numeric_usage(recorder.last.get("usage")),
                    )
                    base["decision_request_count"] = recorder.call_count
                    base["decision_attempted"] = recorder.attempted
                    base["decision_fallback_reason"] = recorder.last.get(
                        "fallback_reason"
                    )
                    base["decision_call_count_valid"] = recorder.call_count == 1
                    if arm == "with_decisions" and recorder.call_count != 1:
                        base.update(
                            status="failed",
                            failure_class="decision_call_count_violation",
                        )
                else:
                    base["decision_request_count"] = 0
                    base["decision_attempted"] = False
                    base["decision_call_count_valid"] = arm == "no_decisions"
                arms[arm] = base
            off, active = arms[ARMS[0]], arms[ARMS[1]]
            equivalence = normalized_em(off.get("answer"), active.get("answer"))
            for record in arms.values():
                record["normalized_arm_equivalence_em"] = equivalence
                record["normalized_reference_exact_match"] = normalized_em(
                    record.get("answer"), frame.reference_answer
                )
                record["normalized_reference_contained_in_answer"] = (
                    normalized_reference_contains(
                        record.get("answer"), frame.reference_answer
                    )
                )
                exact = record["normalized_reference_exact_match"]
                contained = record["normalized_reference_contained_in_answer"]
                record["gold_match_label"] = (
                    "exact_proxy"
                    if exact is True
                    else "contained_proxy"
                    if contained is True
                    else "non_match_proxy"
                    if contained is False
                    else "uncertain"
                )
                record["needs_human_review"] = record[
                    "status"
                ] != "completed" or record["gold_match_label"] in {
                    "non_match_proxy",
                    "uncertain",
                }
                records.append(record)
            if checkpoint is not None:
                checkpoint(records)
            if stop_on_failure and any(
                record.get("status") == "failed" for record in arms.values()
            ):
                return records
    return records


def _live_decision_call(
    model: str, api_key_env: str, *, timeout_seconds: float | None = None
) -> DecisionCall:
    configured_key = get_settings().decisions_api_key
    api_key = os.environ.get(api_key_env) or (
        configured_key.get_secret_value() if configured_key is not None else None
    )
    if not api_key:
        raise SystemExit(
            f"--live requires {api_key_env} or the configured Decisions API key"
        )

    timeout_seconds = (
        get_settings().decisions_timeout_seconds
        if timeout_seconds is None
        else timeout_seconds
    )

    async def call(question: str) -> Mapping[str, Any]:
        try:
            result, elapsed_ms, _ = await post_decision(
                question, model, api_key=api_key, timeout_seconds=timeout_seconds
            )
        except LiveDecisionError as exc:
            raise RuntimeError(exc.reason) from None
        return {
            "status": "ok",
            "route": result.route.value,
            "resolved_model": result.resolved_model,
            "elapsed_ms": elapsed_ms,
            "usage": dict(result.usage) if result.usage is not None else None,
        }

    return call


class _RecordingDecisionProvider:
    def __init__(self, call: DecisionCall) -> None:
        self.call = call
        self.last: Mapping[str, Any] | None = None
        self.call_count = 0
        self.attempted = False

    async def __call__(self, request: Any) -> Mapping[str, Any]:
        self.call_count += 1
        self.attempted = True
        try:
            result = dict(await self.call(request.text))
        except Exception as exc:
            self.last = {
                "status": "fallback",
                "route": "legacy",
                "fallback_reason": type(exc).__name__,
            }
            raise
        self.last = result
        return {
            "model": result.get("resolved_model"),
            "answers": [
                {
                    "type": "choice",
                    "name": "route",
                    "choice": result.get("route"),
                    "probabilities": [],
                }
            ],
            "usage": result.get("usage") or {},
        }


def _build_service_factory(
    model: str,
    api_key_env: str,
    *,
    generation_model: str,
    generation_api_key_env: str | None = None,
) -> ServiceFactory:
    from pydantic import SecretStr

    configured = get_settings()
    generation_key_name = generation_api_key_env or api_key_env
    generation_key = os.environ.get(generation_key_name)
    if not generation_key:
        raise SystemExit(f"--live requires generation credential {generation_key_name}")
    decision_call = _live_decision_call(
        model, api_key_env, timeout_seconds=configured.decisions_timeout_seconds
    )
    supplied_api_key = os.environ.get(api_key_env)

    def factory(mode: str) -> AgentService:
        updates: dict[str, Any] = {
            "decisions_mode": "active" if mode == "with_decisions" else "off",
            "decisions_model": model,
            "openai_model": generation_model,
            "openai_api_key": SecretStr(generation_key),
        }
        if supplied_api_key:
            updates["decisions_api_key"] = SecretStr(supplied_api_key)
        settings: AppSettings = configured.model_copy(update=updates)
        recorder = (
            _RecordingDecisionProvider(decision_call)
            if mode == "with_decisions"
            else None
        )
        service = AgentService(settings=settings, decisions_provider=recorder)
        if recorder is not None:
            service._comparison_decision_recorder = recorder
        return service

    return factory


def pair_summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    for record in records:
        pair = pairs.setdefault(str(record["pair_id"]), {"pair_id": record["pair_id"]})
        pair[str(record["arm"])] = dict(record)
    return list(pairs.values())


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = pair_summaries(records)
    em_pairs = [
        pair
        for pair in pairs
        if pair.get("no_decisions", {}).get("normalized_arm_equivalence_em") is not None
    ]
    by_arm = {
        arm: {
            "records": sum(record.get("arm") == arm for record in records),
            "completed_records": sum(
                record.get("arm") == arm and record.get("status") == "completed"
                for record in records
            ),
            "reference_exact_match": sum(
                record.get("arm") == arm
                and record.get("normalized_reference_exact_match") is True
                for record in records
            ),
            "reference_exact_match_denominator": sum(
                record.get("arm") == arm
                and record.get("normalized_reference_exact_match") is not None
                for record in records
            ),
            "reference_contained_proxy": sum(
                record.get("arm") == arm
                and record.get("normalized_reference_contained_in_answer") is True
                for record in records
            ),
            "decision_request_count": sum(
                record.get("decision_request_count", 0)
                for record in records
                if record.get("arm") == arm
            ),
        }
        for arm in ARMS
    }
    return {
        "pairs": len(pairs),
        "records": len(records),
        "completed_records": sum(
            record.get("status") == "completed" for record in records
        ),
        "citation_absent_records": sum(
            not record.get("citations_present") for record in records
        ),
        "normalized_arm_equivalence_em_pairs": sum(
            pair.get("no_decisions", {}).get("normalized_arm_equivalence_em") is True
            for pair in em_pairs
        ),
        "normalized_arm_equivalence_em_denominator": len(em_pairs),
        "by_arm": by_arm,
        "needs_human_review_pairs": sum(
            pair.get("no_decisions", {}).get("needs_human_review", True)
            or pair.get("with_decisions", {}).get("needs_human_review", True)
            for pair in pairs
        ),
        "warning": (
            "reference metrics are strict automatic proxies only; non-match and "
            "uncertain cases require human review. Arm equivalence is not correctness."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=FRAMES_PATH)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--model", default=os.environ.get("INDIC_EVAL_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument(
        "--generation-model",
        default="gpt-4.1-mini",
        help="Exact direct OpenAI generation model",
    )
    parser.add_argument("--generation-api-key-env", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--live", action="store_true", help="run billable Decisions and agent inference"
    )
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    if not 1 <= args.limit <= MAX_CASES or not 1 <= args.repetitions <= MAX_REPETITIONS:
        raise SystemExit(
            f"--limit must be 1..{MAX_CASES}; "
            f"--repetitions must be 1..{MAX_REPETITIONS}"
        )
    frames = load_frames(args.frames)
    output_dir = args.output_dir or Path("tmp/decisions-live") / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(
        frames,
        model=args.model,
        repetitions=args.repetitions,
        limit=args.limit,
        generation_model=args.generation_model,
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.live:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "manifest": str(output_dir / "manifest.json"),
                    "warning": "no live calls made",
                }
            )
        )
        return 0

    def checkpoint(records: Sequence[Mapping[str, Any]]) -> None:
        temporary = output_dir / "records.jsonl.partial"
        temporary.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        )
        temporary.replace(output_dir / "records.jsonl")

    records = await run_comparison(
        frames,
        service_factory=_build_service_factory(
            args.model,
            args.api_key_env,
            generation_model=args.generation_model,
            generation_api_key_env=args.generation_api_key_env,
        ),
        decision_call=None,
        model=args.model,
        repetitions=args.repetitions,
        limit=args.limit,
        checkpoint=checkpoint,
        stop_on_failure=True,
        generation_model=args.generation_model,
    )
    (output_dir / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )
    (output_dir / "pairs.jsonl").write_text(
        "".join(
            json.dumps(pair, ensure_ascii=False) + "\n"
            for pair in pair_summaries(records)
        )
    )
    summary = summarize_records(records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), **summary}))
    return 1 if any(record.get("status") == "failed" for record in records) else 0


def main() -> int:
    try:
        return asyncio.run(_main_async(_parser().parse_args()))
    except FileNotFoundError as exc:
        print(json.dumps({"error": str(exc), "live_calls": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
