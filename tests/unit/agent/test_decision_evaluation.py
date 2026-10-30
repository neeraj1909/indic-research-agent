from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts" / "evaluate_decisions.py"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def case(case_id: str, expected: str = "research") -> dict[str, object]:
    return {
        "id": case_id,
        "split": "dev",
        "family": "test",
        "language": "english",
        "request": "Find recent Hindi OCR research.",
        "expected_route": expected,
        "requires_research": expected == "research",
        "material_clarification": False,
        "label_status": "draft",
        "reviewed_by": None,
    }


def run_cli(
    tmp_path: Path,
    cases: list[dict[str, object]],
    responses: list[dict[str, object]],
    *extra: str,
) -> dict[str, object]:
    cases_path = tmp_path / "cases.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    output_path = tmp_path / "result.json"
    write_jsonl(cases_path, cases)
    write_jsonl(responses_path, responses)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--offline",
            "--cases",
            str(cases_path),
            "--responses",
            str(responses_path),
            "--output",
            str(output_path),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(output_path.read_text())


def run_cli_raw(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--offline",
            "--cases",
            str(tmp_path / "cases.jsonl"),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )


def test_shadow_uses_explicit_controlled_answer_and_missing_response_falls_back(
    tmp_path: Path,
) -> None:
    result = run_cli(
        tmp_path,
        [case("one"), case("two", "legacy")],
        [
            {
                "id": "one",
                "response": {
                    "source": "controlled",
                    "answers": [
                        {"type": "choice", "name": "route", "choice": "supplied_text"}
                    ],
                },
            }
        ],
        "--mode",
        "shadow",
    )

    records = {record["id"]: record for record in result["records"]}
    assert records["one"]["predicted_route"] == "supplied_text"
    assert records["one"]["predicted_route"] != records["one"]["expected_route"]
    assert records["two"]["predicted_route"] == "legacy"
    assert records["two"]["fallback_reason"] == "invalid_answer"
    assert result["run"]["actual_model_calls"] == 0
    assert result["metrics"]["reviewed_case_count"] == 0


def test_off_makes_zero_controlled_calls_and_records_legacy(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    write_jsonl(cases_path, [case("one")])
    completed = run_cli_raw(tmp_path, "--mode", "off")
    assert completed.returncode == 0, completed.stderr
    result = json.loads(
        (tmp_path / "tmp/decisions-implementation")
        .glob("*.json")
        .__next__()
        .read_text()
    )
    assert result["run"]["controlled_provider_calls"] == 0
    assert result["records"][0]["predicted_route"] == "legacy"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("family", "", "family must be a non-empty string"),
        ("language", "", "language must be a non-empty string"),
        ("requires_research", "true", "requires_research must be a JSON boolean"),
        ("expected_route", "nope", "unsupported expected_route"),
    ],
)
def test_case_validation_rejects_malformed_required_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    row = case("one")
    row[field] = value
    write_jsonl(tmp_path / "cases.jsonl", [row])
    completed = run_cli_raw(tmp_path, "--mode", "off")
    assert completed.returncode == 2
    assert message in completed.stderr


def test_empty_split_is_rejected(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "cases.jsonl", [case("one")])
    completed = run_cli_raw(tmp_path, "--mode", "off", "--split", "heldout")
    assert completed.returncode == 2
    assert "selected zero cases" in completed.stderr


def test_shadow_requires_responses_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "cases.jsonl", [case("one")])
    missing = run_cli_raw(tmp_path, "--mode", "shadow")
    assert missing.returncode == 2
    assert "--responses is required when --mode=shadow" in missing.stderr

    responses = tmp_path / "responses.jsonl"
    write_jsonl(responses, [{"id": "one"}, {"id": "one"}])
    duplicate = run_cli_raw(tmp_path, "--mode", "shadow", "--responses", str(responses))
    assert duplicate.returncode == 2
    assert "duplicate response id" in duplicate.stderr


def test_diagnostics_are_arithmetic_and_model_metadata_is_not_resolved(
    tmp_path: Path,
) -> None:
    rows = [case("research", "research"), case("skip", "research")]
    rows[1]["language"] = "hindi"
    rows[1]["requires_research"] = True
    responses = [
        {
            "id": "research",
            "response": {
                "model": "payload-must-not-become-resolved",
                "answers": [{"type": "choice", "name": "route", "choice": "research"}],
            },
        },
        {
            "id": "skip",
            "response": {
                "answers": [
                    {"type": "choice", "name": "route", "choice": "supplied_text"}
                ],
            },
        },
    ]
    result = run_cli(tmp_path, rows, responses, "--mode", "shadow")
    assert result["run"]["resolved_model"] is None
    assert all(record["resolved_model"] is None for record in result["records"])
    assert result["metrics"]["potential_research_skips"] == 1
    assert (
        result["metrics"]["language_diagnostics"]["hindi"]["potential_research_skips"]
        == 1
    )
    assert result["metrics"]["language_diagnostics"]["english"]["route_errors"] == 0


def test_default_outputs_are_unique_and_do_not_overwrite(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "cases.jsonl", [case("one")])
    first = run_cli_raw(tmp_path, "--mode", "off")
    second = run_cli_raw(tmp_path, "--mode", "off")
    assert first.returncode == second.returncode == 0
    outputs = sorted((tmp_path / "tmp/decisions-implementation").glob("*.json"))
    assert len(outputs) == 2
    assert outputs[0].read_text() != ""


def test_unknown_response_id_is_a_helpful_error(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    output_path = tmp_path / "result.json"
    write_jsonl(cases_path, [case("one")])
    write_jsonl(responses_path, [{"id": "unknown", "response": {}}])
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--offline",
            "--cases",
            str(cases_path),
            "--responses",
            str(responses_path),
            "--output",
            str(output_path),
            "--mode",
            "shadow",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "not present in cases" in completed.stderr
