"""Export the compiled LangGraph topology for documentation."""

from __future__ import annotations

import argparse
from pathlib import Path

from indic_research_agent.agent.graph import build_agent_graph
from indic_research_agent.tools.search import SearchTool


class TopologyModel:
    """Compile the graph without calling a model or a research provider."""

    def bind_tools(self, tools: list[object]) -> TopologyModel:
        return self


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/diagrams/langgraph-agent.png"),
        help="PNG path relative to the repository root unless absolute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    output_path = args.output
    if not output_path.is_absolute():
        output_path = repository_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph = build_agent_graph(TopologyModel(), search_tool=SearchTool())
    output_path.write_bytes(graph.get_graph().draw_mermaid_png())
    try:
        display_path = output_path.relative_to(repository_root)
    except ValueError:
        display_path = output_path
    print(f"Wrote {display_path}")


if __name__ == "__main__":
    main()
