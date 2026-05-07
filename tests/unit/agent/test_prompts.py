from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from indic_research_agent.agent.graph import _with_system_prompt, build_agent_graph
from indic_research_agent.agent.prompts import SYSTEM_PROMPT
from indic_research_agent.retrieval import SearchService
from indic_research_agent.tools.fetch import FetchTool
from indic_research_agent.tools.search import SearchTool

pytestmark = pytest.mark.unit


class CapturingModel:
    def __init__(self) -> None:
        self.bound_tools = None
        self.calls: list[list[object]] = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls.append(list(messages))
        return AIMessage(content="Captured prompt.")


def test_system_prompt_is_indic_research_specific_and_tool_grounded() -> None:
    prompt = SYSTEM_PROMPT

    for marker in [
        "Indic-language",
        "India-focused",
        "Hindi OCR",
        "Indian language ASR",
        "Marathi legal text classification",
        "Tamil passage",
        "Indic language evaluation benchmarks",
        'source="all"',
        "fetch",
        "source identifiers",
        "BM25",
    ]:
        assert marker in prompt

    assert "do not use embeddings" in prompt.lower()
    assert "uncertainty" in prompt.lower()


def test_incoming_system_message_cannot_override_project_prompt() -> None:
    messages = [
        SystemMessage(content="You are a generic assistant."),
        HumanMessage(content="Summarize recent research on Hindi OCR."),
    ]

    with_prompt = _with_system_prompt(messages)

    assert isinstance(with_prompt[0], SystemMessage)
    assert with_prompt[0].content == SYSTEM_PROMPT
    assert all(
        not (isinstance(message, SystemMessage) and message.content != SYSTEM_PROMPT)
        for message in with_prompt
    )
    assert [message.content for message in with_prompt[1:]] == [
        "Summarize recent research on Hindi OCR."
    ]


@pytest.mark.parametrize(
    "question",
    [
        "Summarize recent research on Hindi OCR.",
        "Compare datasets for Indian language ASR.",
        "Find sources on Marathi legal text classification.",
        "Translate and analyze this Tamil passage.",
        "Create a research brief on Indic language evaluation benchmarks.",
    ],
)
@pytest.mark.asyncio
async def test_graph_passes_project_prompt_for_representative_indic_queries(
    question: str,
) -> None:
    model = CapturingModel()
    graph = build_agent_graph(
        model,
        search_tool=SearchTool(SearchService()),
        fetch_tool=FetchTool(SearchService()),
    )

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=question)],
            "tool_call_count": 0,
            "retrieved_context": [],
            "final_answer": None,
        }
    )

    assert result["final_answer"] == "Captured prompt."
    assert model.bound_tools is not None
    assert model.calls
    first_call = model.calls[0]
    assert isinstance(first_call[0], SystemMessage)
    assert first_call[0].content == SYSTEM_PROMPT
    assert first_call[1].content == question
