from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

pytestmark = pytest.mark.unit

BANNED_DIRECT_DEPENDENCIES = {
    "chromadb",
    "faiss",
    "faiss-cpu",
    "faiss-gpu",
    "milvus",
    "pinecone",
    "pinecone-client",
    "pgvector",
    "pymilvus",
    "sentence-transformers",
    "weaviate-client",
}


def test_project_does_not_directly_depend_on_vector_or_embedding_packages() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    dependency_names = {
        Requirement(dependency).name.casefold()
        for dependency in _direct_dependencies(pyproject)
    }

    assert dependency_names.isdisjoint(BANNED_DIRECT_DEPENDENCIES)


def _direct_dependencies(pyproject: dict) -> list[str]:
    dependencies = list(pyproject["project"].get("dependencies", []))
    for group in pyproject.get("dependency-groups", {}).values():
        dependencies.extend(group)
    return dependencies
