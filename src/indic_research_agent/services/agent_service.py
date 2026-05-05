"""High-level agent service."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from indic_research_agent.agent.graph import build_agent_graph
from indic_research_agent.agent.llm import create_chat_model
from indic_research_agent.retrieval import create_seed_search_service
from indic_research_agent.services.querykit_service import QueryKitService
from indic_research_agent.tools.fetch import FetchTool
from indic_research_agent.tools.search import SearchTool


@dataclass(frozen=True)
class AgentAnswer:
    answer: str
    retrieved_context: list[str]
    tool_call_count: int


class AgentService:
    """Application-facing wrapper around the LangGraph agent."""

    def __init__(self, graph=None) -> None:
        self._graph = graph if graph is not None else create_default_graph()

    async def answer(self, question: str) -> AgentAnswer:
        if not question.strip():
            raise ValueError("question is required")
        state = await self._graph.ainvoke(
            {
                "messages": [HumanMessage(content=question)],
                "tool_call_count": 0,
                "retrieved_context": [],
                "final_answer": None,
            }
        )
        return AgentAnswer(
            answer=state.get("final_answer") or str(state["messages"][-1].content),
            retrieved_context=list(state.get("retrieved_context", [])),
            tool_call_count=int(state.get("tool_call_count", 0)),
        )


def create_default_graph():
    search_service = create_seed_search_service()
    search_tool = SearchTool(search_service, QueryKitService())
    fetch_tool = FetchTool(search_service)
    return build_agent_graph(
        create_chat_model(),
        search_tool=search_tool,
        fetch_tool=fetch_tool,
    )
