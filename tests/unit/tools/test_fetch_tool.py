from __future__ import annotations

import pytest

from indic_research_agent.retrieval import DocumentChunk, SearchService
from indic_research_agent.tools.fetch import FetchTool
from indic_research_agent.tools.schemas import FetchToolInput

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_fetch_tool_returns_bounded_local_chunk() -> None:
    tool = FetchTool(
        SearchService(
            [
                DocumentChunk(
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    title="Title",
                    source="test://doc",
                    text="abcdef",
                    metadata={"kind": "test"},
                )
            ]
        )
    )

    result = await tool.run(
        FetchToolInput(document_id="doc-1", chunk_id="chunk-1", max_chars=3)
    )

    assert result.content == "abc"
    assert result.metadata == {"kind": "test"}


@pytest.mark.asyncio
async def test_fetch_tool_fails_clearly_for_missing_chunk() -> None:
    tool = FetchTool(SearchService())

    with pytest.raises(ValueError, match="was not found"):
        await tool.run(FetchToolInput(document_id="missing"))
