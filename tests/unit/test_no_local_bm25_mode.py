from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from pydantic import ValidationError

from indic_research_agent.tools.schemas import SearchToolInput

pytestmark = pytest.mark.unit

FORBIDDEN_SOURCE_PATTERNS = (
    "indic_research_agent." + "retrieval",
    "from indic_research_agent." + "retrieval",
    "import " + "bm" + "25s",
    "Fetch" + "Tool",
    "Fetch" + "Tool" + "Input",
    "fetch.local_" + "bm" + "25",
    "Document" + "Service",
    "Document" + "Repository",
    "Document" + "Chunk" + "Input",
    "document_" + "service",
    "source=" + '"local"',
    "source=" + '"all"',
)


def test_search_tool_input_has_no_local_or_all_source_mode() -> None:
    assert "source" not in SearchToolInput.model_fields

    with pytest.raises(ValidationError):
        SearchToolInput(query="Hindi OCR", **{"source": "local"})

    with pytest.raises(ValidationError):
        SearchToolInput(query="Hindi OCR", **{"source": "all"})


def test_first_party_runtime_code_does_not_reference_removed_local_mode() -> None:
    scanned_files = [
        path
        for root in (Path("src"), Path("scripts"))
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]

    offenders: list[str] = []
    for path in scanned_files:
        content = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_SOURCE_PATTERNS:
            if pattern in content:
                offenders.append(f"{path}:{pattern}")

    assert offenders == []


def test_project_does_not_directly_depend_on_removed_keyword_package() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    dependency_names = {
        Requirement(dependency).name.casefold()
        for dependency in pyproject["project"].get("dependencies", [])
    }

    assert "bm" + "25s" not in dependency_names
