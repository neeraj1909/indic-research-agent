"""LiteLLM chat model factory."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence

from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM

from indic_research_agent.config import AppSettings, get_settings


def create_chat_model(settings: AppSettings | None = None) -> ChatLiteLLM:
    """Create the LiteLLM-backed LangChain chat model."""

    settings = settings or get_settings()
    api_key = (
        settings.litellm_api_key.get_secret_value()
        if settings.litellm_api_key is not None
        else None
    )
    return ChatLiteLLM(
        model=settings.litellm_model,
        custom_llm_provider=settings.litellm_custom_llm_provider,
        api_key=api_key,
        api_base=settings.litellm_api_base,
        temperature=settings.litellm_temperature,
        request_timeout=settings.litellm_timeout_seconds,
        max_tokens=settings.litellm_max_tokens,
        streaming=settings.litellm_streaming,
    )


async def smoke_chat(prompt: str, settings: AppSettings | None = None) -> str:
    """Run one LiteLLM-backed chat completion."""

    model = create_chat_model(settings)
    response = await model.ainvoke([HumanMessage(content=prompt)])
    return str(response.content)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LiteLLM adapter utilities")
    parser.add_argument("--smoke", action="store_true", help="Run adapter smoke check")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the configured LiteLLM provider instead of dry-running",
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
    print(json.dumps(settings.masked_litellm_config, sort_keys=True))
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
