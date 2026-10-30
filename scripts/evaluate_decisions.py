#!/usr/bin/env python3
"""Evaluate the offline route-decision contract with explicit local inputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from indic_research_agent.agent.decision import (
    MAX_REQUEST_CHARS,
    DecisionRequest,
    DecisionRoute,
    decide,
)

MODEL = "gpt-luna-6.1"
RUBRIC = "route-v1"
ROUTES = tuple(route.value for route in DecisionRoute)
CASE_FIELDS = {
    "id",
    "split",
    "family",
    "language",
    "request",
    "expected_route",
    "requires_research",
    "material_clarification",
    "label_status",
    "reviewed_by",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--offline", action="store_true", required=True)
    result.add_argument("--cases", type=Path, required=True)
    result.add_argument("--responses", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument(
        "--split", choices=("dev", "development", "heldout", "all"), default="all"
    )
    result.add_argument("--mode", choices=("off", "shadow"), default="off")
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = read_jsonl(path)
    seen: set[str] = set()
    for number, case in enumerate(cases, 1):
        missing = sorted(CASE_FIELDS - case.keys())
        if missing:
            raise ValueError(
                f"{path}:{number}: missing required fields: {', '.join(missing)}"
            )
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{path}:{number}: id must be a non-empty string")
        if case_id in seen:
            raise ValueError(f"{path}:{number}: duplicate case id: {case_id}")
        seen.add(case_id)
        if case["split"] not in {"dev", "development", "heldout"}:
            raise ValueError(f"{path}:{number}: unsupported split: {case['split']!r}")
        case["split"] = "development" if case["split"] == "dev" else case["split"]
        for field in ("family", "language"):
            if not isinstance(case[field], str) or not case[field].strip():
                raise ValueError(f"{path}:{number}: {field} must be a non-empty string")
        if not isinstance(case["request"], str):
            raise ValueError(f"{path}:{number}: request must be a string")
        if len(case["request"]) > MAX_REQUEST_CHARS:
            raise ValueError(
                f"{path}:{number}: request must be at most "
                f"{MAX_REQUEST_CHARS} characters"
            )
        if case["expected_route"] not in ROUTES:
            raise ValueError(f"{path}:{number}: unsupported expected_route")
        for field in ("requires_research", "material_clarification"):
            if type(case[field]) is not bool:
                raise ValueError(f"{path}:{number}: {field} must be a JSON boolean")
        if case["label_status"] not in {"draft", "reviewed"}:
            raise ValueError(f"{path}:{number}: label_status must be draft or reviewed")
        if case["label_status"] == "reviewed" and (
            not isinstance(case["reviewed_by"], str) or not case["reviewed_by"]
        ):
            raise ValueError(f"{path}:{number}: reviewed cases require reviewed_by")
    return cases


def load_responses(path: Path, case_ids: set[str]) -> dict[str, object]:
    responses: dict[str, object] = {}
    for number, row in enumerate(read_jsonl(path), 1):
        case_id = row.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{path}:{number}: response id must be a non-empty string")
        if case_id not in case_ids:
            raise ValueError(
                f"{path}:{number}: response id is not present in cases: {case_id}"
            )
        if case_id in responses:
            raise ValueError(f"{path}:{number}: duplicate response id: {case_id}")
        responses[case_id] = row.get("response")
    return responses


def selected(cases: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return cases
    if split == "dev":
        split = "development"
    return [case for case in cases if case["split"] == split]


async def evaluate(
    cases: list[dict[str, Any]], responses: dict[str, object], mode: str
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    calls = 0
    for case in cases:
        request = DecisionRequest(case["request"], MODEL, RUBRIC)
        started = time.perf_counter()
        if mode == "off":
            outcome = None
        else:
            calls += 1

            async def provider(_request: DecisionRequest, case_id: str = case["id"]):
                return responses.get(case_id, {})

            outcome = await decide(request, provider)
        record = {
            "id": case["id"],
            "split": case["split"],
            "language": case["language"],
            "expected_route": case["expected_route"],
            "predicted_route": DecisionRoute.LEGACY.value
            if outcome is None
            else outcome.route.value,
            "status": "off" if outcome is None else outcome.status,
            "fallback_reason": None if outcome is None else outcome.fallback_reason,
            "source": None if outcome is None else outcome.source,
            "requested_model": MODEL,
            "resolved_model": None,
            "rubric_version": RUBRIC,
            "evaluation_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        records.append(record)
    return records, calls


def metrics(
    cases: list[dict[str, Any]], records: list[dict[str, Any]], calls: int, mode: str
) -> dict[str, Any]:
    by_id = {case["id"]: case for case in cases}
    confusion = {
        expected: {predicted: 0 for predicted in ROUTES} for expected in ROUTES
    }
    for record in records:
        confusion[record["expected_route"]][record["predicted_route"]] += 1
    f1_scores = []
    for route in ROUTES:
        tp = confusion[route][route]
        fp = sum(confusion[expected][route] for expected in ROUTES if expected != route)
        fn = sum(
            confusion[route][predicted] for predicted in ROUTES if predicted != route
        )
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1_scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0
        )
    potential_skips = sum(
        by_id[r["id"]]["requires_research"]
        and r["predicted_route"] in {"supplied_text", "unsupported_local_documents"}
        for r in records
    )
    clarification_misses = sum(
        by_id[r["id"]]["material_clarification"] and r["predicted_route"] != "clarify"
        for r in records
    )
    language_diagnostics: dict[str, dict[str, int]] = {}
    for record in records:
        language = record["language"]
        diagnostics = language_diagnostics.setdefault(
            language,
            {
                "case_count": 0,
                "route_errors": 0,
                "potential_research_skips": 0,
                "clarification_misses": 0,
            },
        )
        diagnostics["case_count"] += 1
        diagnostics["route_errors"] += record["status"] == "fallback"
        case_data = by_id[record["id"]]
        diagnostics["potential_research_skips"] += case_data[
            "requires_research"
        ] and record["predicted_route"] in {
            "supplied_text",
            "unsupported_local_documents",
        }
        diagnostics["clarification_misses"] += (
            case_data["material_clarification"]
            and record["predicted_route"] != "clarify"
        )
    return {
        "case_count": len(records),
        "calls": calls,
        "mode": mode,
        "baseline": {
            "route": "legacy",
            "meaning": "passthrough route, not observed model decisions",
        },
        "reviewed_case_count": sum(
            case["label_status"] == "reviewed" for case in cases
        ),
        "draft_evidence_warning": (
            "Controlled/draft evidence; genuinely reviewed cases and "
            "actual model calls: 0."
            if not any(case["label_status"] == "reviewed" for case in cases)
            else "Controlled responses are wiring evidence, not actual model quality."
        ),
        "confusion": confusion,
        "macro_f1": round(sum(f1_scores) / len(f1_scores), 6),
        "macro_f1_routes": list(ROUTES),
        "potential_research_skips": potential_skips,
        "unsafe_research_skips": potential_skips,
        "unsafe_research_skip_definition": (
            "HYPOTHETICAL: requires_research=true and prediction is supplied_text "
            "or unsupported_local_documents, routes that would bypass search under "
            "the contemplated semantics; clarify defers search and legacy is unknown. "
            "No action occurred in this offline evaluator."
        ),
        "material_clarification_misses": clarification_misses,
        "by_language": dict(Counter(record["language"] for record in records)),
        "language_diagnostics": language_diagnostics,
        "by_split": dict(Counter(record["split"] for record in records)),
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.offline:
        parser().error("--offline is required; this evaluator has no live mode")
    if args.mode == "shadow" and args.responses is None:
        parser().error("--responses is required when --mode=shadow")
    try:
        cases = load_cases(args.cases)
        all_case_ids = {case["id"] for case in cases}
        responses = (
            load_responses(args.responses, all_case_ids)
            if args.mode == "shadow"
            else {}
        )
        selected_split = "development" if args.split == "dev" else args.split
        cases = selected(cases, selected_split)
        if not cases:
            raise ValueError(f"split {args.split!r} selected zero cases")
        records, calls = asyncio.run(evaluate(cases, responses, args.mode))
        payload = {
            "run": {
                "created_at": datetime.now(UTC).isoformat(),
                "mode": args.mode,
                "split": selected_split,
                "offline": True,
                "requested_model": MODEL,
                "resolved_model": None,
                "rubric_version": RUBRIC,
                "actual_model_calls": 0,
                "controlled_provider_calls": calls,
            },
            "metrics": metrics(cases, records, calls, args.mode),
            "records": records,
        }
    except ValueError as exc:
        print(f"evaluate_decisions: {exc}", file=sys.stderr)
        return 2
    if args.output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        output = Path("tmp/decisions-implementation") / f"replay-{stamp}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        except FileExistsError:
            print(
                f"evaluate_decisions: default output already exists: {output}",
                file=sys.stderr,
            )
            return 2
    else:
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps({"output": str(output), "case_count": len(records), "calls": calls})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
