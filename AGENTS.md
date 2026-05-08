# indic-research-agent Instructions

## Active PRP

- Plan file: `/home/neeraj/prp-plans/indic-research-agent/2026-05-08-query-kit-adapter-verification-observability-plan.md`
- When asked to continue implementation, re-open the plan first and execute the earliest incomplete checkbox in `## Implementation Blueprint`.
- After each meaningful work chunk, update the plan's Status Snapshot, checklist state, blockers, and `## What else remains?`.
- Do not run whole-app Docker redeploy, Chainlit browser e2e, or Phoenix e2e for this PRP until the plan's standalone query-kit isolation gates pass outside Docker.

## Architecture Constraints

- Use LiteLLM as the Phase 1 LLM layer.
- Use query-kit from `https://github.com/neeraj1909/query-kit`, imported as `query_cli`.
- Retrieval must be BM25/keyword-based first.
- Do not add embeddings, vector databases, `pgvector`, FAISS, Chroma, Milvus, Weaviate, Pinecone, sentence-transformers, or embedding API calls unless a later PRP records evidence that BM25 is insufficient.
- Keep Chainlit as a thin UI adapter. Put orchestration in services and LangGraph modules.
- Keep retrieval logic testable without an LLM.

## Isolation-First Debug Protocol

- For query-kit/provider bugs, debug in this order only: direct provider API contract with `curl`/`jq`/official docs, query-kit itself outside Docker, this app's adapter/unit probes, then Docker/Chainlit/Phoenix e2e.
- Semantic Scholar work starts from the live docs and swagger: `curl -fsS https://api.semanticscholar.org/graph/v1/swagger.json | jq '.paths["/paper/search"].parameters'` before changing adapter code.
- Treat `/home/neeraj/Code/query-kit` as an independent component imported as `query_cli`. Before app-wrapper, compose, or e2e debugging, prove the component with commands such as `cd /home/neeraj/Code/query-kit && uv run pytest` and `cd /home/neeraj/Code/query-kit && uv run query-cli search "Hindi OCR" --provider semantic-scholar --limit 3 --format json`.
- If many cascading bugs appear or more than one component fails at once, pause instead of continuing. Write a working-memory file under `tmp/<prp-slug>/WORKING_MEMORY.md` with goal, repo status, commands run, failure taxonomy, blockers, and the next isolated debug loop. Then stop and tell the developer that the coding agent will give better results if the clean work is saved with an approved git commit and the failing component is debugged/fixed/verified in isolation before returning.

## Validation Commands

- Format/lint: `uv run ruff format . && uv run ruff check --fix .`
- Unit tests: `timeout 60 uv run pytest -m unit --no-cov`
- Integration tests: `uv run pytest -m integration`
- End-to-end smoke: `uv run pytest -m e2e` or `uv run python scripts/smoke_query.py`

