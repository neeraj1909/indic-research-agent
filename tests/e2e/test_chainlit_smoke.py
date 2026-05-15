from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def test_smoke_query_cli_exercises_agent_stack() -> None:
    database_url = os.environ.get("APP_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not database_url or not os.environ.get("REDIS_URL"):
        pytest.skip(
            "APP_DATABASE_URL or DATABASE_URL, plus REDIS_URL, are required for "
            "e2e smoke tests"
        )

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_query.py",
            "--question",
            "Find public research on Hindi OCR for this smoke run.",
            "--cache-ttl-seconds",
            "60",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["answer"]
    assert payload["source_ids"]
    assert payload["tool_call_count"] == 1
    assert payload["persisted_tool_calls"] >= 1
    assert payload["cache_hits"] >= 1
    assert payload["cache_namespaces"] == ["tool.search"]
