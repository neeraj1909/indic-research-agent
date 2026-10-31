"""Direct OpenAI chat model factory."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence

from langchain_core.messages import HumanMessage

from indic_research_agent.agent.openai_streaming import (
    OpenAICompatibleStreamingChatModel,
)
from indic_research_agent.config import AppSettings, get_settings


def create_chat_model(
    settings: AppSettings | None = None,
) -> OpenAICompatibleStreamingChatModel:
    """Create the direct OpenAI model; never inherit a proxy base URL."""
    settings = settings or get_settings()
    if (
        settings.openai_api_key is None
        or not settings.openai_api_key.get_secret_value()
    ):
        raise ValueError("OPENAI_API_KEY is required")
    return OpenAICompatibleStreamingChatModel(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        api_base="https://api.openai.com/v1",
        temperature=settings.openai_temperature,
        request_timeout=settings.openai_timeout_seconds,
        max_tokens=settings.openai_max_tokens,
        streaming=settings.openai_streaming,
    )


async def smoke_chat(prompt: str, settings: AppSettings | None = None) -> str:
    """Run one OpenAI-backed chat completion."""

    model = create_chat_model(settings)
    response = await model.ainvoke([HumanMessage(content=prompt)])
    return str(response.content)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI adapter utilities")
    parser.add_argument("--smoke", action="store_true", help="Run adapter smoke check")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the configured OpenAI provider instead of dry-running",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with the word ready.",
        help="Prompt used for --smoke --live",
    )
    return parser


async def _main_async(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()

    if not args.smoke:
        parser.print_help()
        return 0

    model = create_chat_model(settings)
    print(
        json.dumps(
            {
                "model": settings.openai_model,
                "has_api_key": settings.openai_api_key is not None,
            },
            sort_keys=True,
        )
    )
    print(f"adapter={model.__class__.__module__}.{model.__class__.__name__}")

    if not args.live:
        print("dry_run=true")
        return 0

    content = await smoke_chat(args.prompt, settings)
    print(content)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main())
