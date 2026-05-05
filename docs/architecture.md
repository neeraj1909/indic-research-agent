# Architecture

`indic-research-agent` is a BM25-first research agent. The first working slice is
designed so retrieval, tool execution, agent orchestration, and UI can be tested
separately.

## Runtime Flow

```text
Chainlit UI
  -> ChatController
  -> AgentService
  -> LangGraph graph
  -> LiteLLM chat model
  -> SearchTool / FetchTool
  -> SearchService / QueryKitService
  -> BM25 index / query-kit providers
```

## Module Map

| Path | Responsibility |
| --- | --- |
| `src/indic_research_agent/config.py` | Environment-backed settings. |
| `src/indic_research_agent/ui/` | Chainlit adapter only. |
| `src/indic_research_agent/controllers/` | UI-facing workflow controllers. |
| `src/indic_research_agent/services/` | Application orchestration and external service adapters. |
| `src/indic_research_agent/agent/` | LangGraph state, prompts, model factory, graph construction. |
| `src/indic_research_agent/tools/` | Typed Search and Fetch tool implementations. |
| `src/indic_research_agent/retrieval/` | BM25 tokenizer, index, search contracts, and seed corpus. |
| `src/indic_research_agent/models/` | SQLAlchemy models, added in Phase 3. |
| `src/indic_research_agent/repositories/` | Persistence repositories, added in Phase 3. |

## Retrieval Policy

The project must not use embeddings or vector databases in the initial
implementation. Retrieval quality should be improved with:

- BM25 tokenization and chunking choices.
- Field weighting and filtering.
- Query rewriting or expansion when logged and testable.
- Cache-aware Search and Fetch tools.

The direct dependency guard in `tests/unit/test_no_vector_dependencies.py`
prevents common vector and embedding packages from being added directly.

## Current Limits

- Phase 1 uses an in-memory seed corpus.
- Query-kit public provider calls are wrapped but not cached until Phase 3.
- Fetch currently supports local chunks by `document_id` and `chunk_id`.
- Live LiteLLM calls require provider environment variables.

## Commands

```bash
uv sync
uv run python -m indic_research_agent.agent.llm --smoke
uv run ruff format . && uv run ruff check --fix .
timeout 60 uv run pytest -m unit --no-cov
uv run chainlit run src/indic_research_agent/ui/chainlit_app.py -w
```

