# indic-research-agent Instructions

## Active PRP

- Plan file: `/home/neeraj/prp-plans/indic-research-agent/2026-05-05-chainlit-streaming-persistence-auth-plan.md`
- When asked to continue implementation, re-open the plan first and execute the earliest incomplete checkbox in `## Implementation Blueprint`.
- After each meaningful work chunk, update the plan's Status Snapshot, checklist state, blockers, and `## What else remains?`.

## Architecture Constraints

- Use LiteLLM as the Phase 1 LLM layer.
- Use query-kit from `https://github.com/neeraj1909/query-kit`, imported as `query_cli`.
- Retrieval must be BM25/keyword-based first.
- Do not add embeddings, vector databases, `pgvector`, FAISS, Chroma, Milvus, Weaviate, Pinecone, sentence-transformers, or embedding API calls unless a later PRP records evidence that BM25 is insufficient.
- Keep Chainlit as a thin UI adapter. Put orchestration in services and LangGraph modules.
- Keep retrieval logic testable without an LLM.

## Validation Commands

- Format/lint: `uv run ruff format . && uv run ruff check --fix .`
- Unit tests: `timeout 60 uv run pytest -m unit --no-cov`
- Integration tests: `uv run pytest -m integration`
- End-to-end smoke: `uv run pytest -m e2e` or `uv run python scripts/smoke_query.py`

