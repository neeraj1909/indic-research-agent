#!/usr/bin/env python3
"""Run one explicitly opted-in live OpenAI Decisions evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from indic_research_agent.agent.decisions_live import (
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    RUBRIC_VERSION,
    LiveDecisionError,
    post_decision,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--live", action="store_true", required=True)
    result.add_argument("--case-id", required=True)
    result.add_argument("--cases", type=Path, required=True)
    result.add_argument("--model", choices=(DEFAULT_MODEL,), default=DEFAULT_MODEL)
    result.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    result.add_argument("--output", type=Path, required=True)
    return result


def load_case(path: Path, case_id: str) -> dict[str, object]:
    matches = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read cases: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"cases:{line_number}: invalid JSON") from exc
        if not isinstance(case, dict):
            raise ValueError(f"cases:{line_number}: expected an object")
        if case.get("id") == case_id:
            matches.append(case)
    if len(matches) != 1:
        raise ValueError(f"case id {case_id!r} must select exactly one case")
    request = matches[0].get("request")
    if not isinstance(request, str) or not request:
        raise ValueError("selected case request must be a non-empty string")
    return matches[0]


def write_output(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


async def _main_async(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        case = load_case(args.cases, args.case_id)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LiveDecisionError("missing_api_key")
        result, latency_ms, status_code = await post_decision(
            case["request"],
            args.model,
            api_key=api_key,
            timeout_seconds=args.timeout,
        )
        payload = {
            "run": {
                "created_at": datetime.now(UTC).isoformat(),
                "status": "ok",
                "route": result.route.value,
                "case_id": args.case_id,
                "requested_model": result.requested_model,
                "resolved_model": result.resolved_model,
                "rubric_version": RUBRIC_VERSION,
                "route_endpoint": "/v1/decisions",
                "http_status": status_code,
                "latency_ms": latency_ms,
                "usage": result.usage,
                "request_count": 1,
            }
        }
    except (LiveDecisionError, ValueError) as exc:
        error = (
            exc
            if isinstance(exc, LiveDecisionError)
            else LiveDecisionError("invalid_input")
        )
        payload = {
            "run": {
                "created_at": datetime.now(UTC).isoformat(),
                "status": "error",
                "error": error.reason,
                "case_id": args.case_id,
                "requested_model": args.model,
                "resolved_model": None,
                "http_status": error.status_code,
                "route_endpoint": "/v1/decisions",
                "request_count": 0
                if error.reason in {"missing_api_key", "invalid_input"}
                else 1,
            }
        }
        write_output(args.output, payload)
        print(f"evaluate_decisions_live: {error.reason}", file=sys.stderr)
        return 1
    write_output(args.output, payload)
    print(json.dumps({"output": str(args.output), "status": "ok", "request_count": 1}))
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main())
