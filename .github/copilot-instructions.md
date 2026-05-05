# Repository Instructions

This project implements a BM25-first research agent using LiteLLM, query-kit, LangGraph, Chainlit, PostgreSQL, Redis, SQLAlchemy, and Docker.

Follow the active PRP at `/home/neeraj/prp-plans/indic-research-agent/2026-05-05-bm25-query-kit-langgraph-agent-plan.md`.

Critical constraints:

- Use LiteLLM, not LightLLM.
- Use `query-kit` from `https://github.com/neeraj1909/query-kit`; import package is `query_cli`.
- Do not use embeddings or vector databases.
- Keep UI, services, retrieval, agent graph, persistence, and tools separated.
- Run `uv run ruff format . && uv run ruff check --fix .` and `timeout 60 uv run pytest -m unit --no-cov` before handoff when feasible.

