"""Probe query-kit providers through the app adapter.

This script does not require Docker, Chainlit, or an LLM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any
from uuid import uuid4

from indic_research_agent.config import AppSettings
from indic_research_agent.services.phoenix_tracing import (
    configure_phoenix,
    force_flush_traces,
    session_attributes,
    set_span_output,
    trace_span,
)
from indic_research_agent.services.querykit_service import QueryKitService

DEFAULT_USER_AGENT = (
    "indic-research-agent/0.1 (+https://github.com/neeraj1909/indic-research-agent)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="Hindi OCR",
        help="Provider query text to run through QueryKitService.",
    )
    parser.add_argument(
        "--providers",
        default=None,
        help=(
            "Comma-separated provider IDs. Defaults to AppSettings/"
            "QUERY_KIT_PROVIDERS. Example: "
            "semantic-scholar,semantic-scholar-web,pubmed,arxiv-web"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of normalized app results to return.",
    )
    parser.add_argument(
        "--since-year",
        type=int,
        default=None,
        help="Optional publication year lower bound.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override QUERY_KIT_TIMEOUT_SECONDS for this probe.",
    )
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("QUERY_CLI_USER_AGENT", DEFAULT_USER_AGENT),
        help="Project-specific User-Agent passed to query-kit HTTP providers.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional trace/session correlation id for Phoenix debugging.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Python logging level for provider diagnostics.",
    )
    return parser.parse_args()


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.user_agent:
        os.environ["QUERY_CLI_USER_AGENT"] = args.user_agent

    settings = AppSettings()
    updates: dict[str, Any] = {}
    if args.providers:
        updates["query_kit_providers"] = args.providers
    if args.timeout is not None:
        updates["query_kit_timeout_seconds"] = args.timeout
    if updates:
        settings = settings.model_copy(update=updates)

    provider_ids = _provider_ids(args.providers) or settings.query_kit_provider_ids
    session_id = args.session_id or f"query-kit-probe:{uuid4().hex[:12]}"
    start = time.perf_counter()
    service = QueryKitService(settings, environ=os.environ)
    with trace_span(
        "probe.query_kit",
        kind="RETRIEVER",
        input_value={
            "query": args.query,
            "providers": provider_ids,
            "limit": args.limit,
            "since_year": args.since_year,
        },
        attributes={
            **session_attributes(
                session_id=session_id,
                metadata={"script": "scripts/probe_querykit_providers.py"},
                tags=["query-kit", "probe"],
            ),
            "query_kit.probe": True,
            "query_kit.query": args.query,
            "query_kit.providers": ",".join(provider_ids),
        },
    ) as probe_span:
        results = await service.search(
            args.query,
            providers=provider_ids,
            limit=args.limit,
            since_year=args.since_year,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        result_payloads = [_result_payload(result) for result in results]
        payload = {
            "query": args.query,
            "providers": provider_ids,
            "limit": args.limit,
            "since_year": args.since_year,
            "timeout_seconds": settings.query_kit_timeout_seconds,
            "user_agent": os.environ.get("QUERY_CLI_USER_AGENT"),
            "session_id": session_id,
            "elapsed_ms": round(elapsed_ms, 1),
            "result_count": len(result_payloads),
            "sources": sorted({item["provider"] for item in result_payloads}),
            "results": result_payloads,
        }
        set_span_output(
            probe_span,
            {
                "elapsed_ms": payload["elapsed_ms"],
                "result_count": payload["result_count"],
                "sources": payload["sources"],
            },
        )
        return payload


def _provider_ids(value: str | None) -> list[str]:
    if not value:
        return []
    return [provider.strip() for provider in value.split(",") if provider.strip()]


def _result_payload(result: Any) -> dict[str, Any]:
    metadata = dict(getattr(result, "metadata", {}) or {})
    snippet = str(getattr(result, "snippet", "") or "")
    return {
        "document_id": getattr(result, "document_id", None),
        "chunk_id": getattr(result, "chunk_id", None),
        "title": getattr(result, "title", None),
        "source": getattr(result, "source", None),
        "provider": metadata.get("provider"),
        "year": metadata.get("year"),
        "venue": metadata.get("venue"),
        "authors": metadata.get("authors", []),
        "snippet_chars": len(snippet),
        "snippet": snippet,
        "metadata": metadata,
    }


async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s %(message)s",
    )
    configure_phoenix()
    payload = await run_probe(args)
    force_flush_traces()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
