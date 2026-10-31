from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts" / "evaluate_decisions_live.py"


def test_live_cli_reports_missing_key_without_network_or_input(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    output = tmp_path / "result.json"
    cases.write_text(json.dumps({"id": "one", "request": "private prompt"}) + "\n")
    environment = {
        name: os.environ[name] for name in ("PATH", "PYTHONPATH") if name in os.environ
    }
    network_guard = (
        "import runpy, socket\n"
        "def block_network(*args, **kwargs):\n"
        "    raise AssertionError('network access blocked')\n"
        "socket.create_connection = block_network\n"
        "socket.socket.connect = block_network\n"
        "socket.socket.connect_ex = block_network\n"
        "import sys\n"
        "sys.argv = sys.argv[1:]\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            network_guard,
            str(SCRIPT),
            "--live",
            "--case-id",
            "one",
            "--cases",
            str(cases),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        cwd=tmp_path,
    )

    assert completed.returncode == 1
    payload = json.loads(output.read_text())
    assert payload["run"]["error"] == "missing_api_key"
    assert payload["run"]["requested_model"] == "gpt-6-luna"
    assert payload["run"]["request_count"] == 0
    assert "private prompt" not in output.read_text()
    assert "OPENAI_API_KEY" not in output.read_text()
