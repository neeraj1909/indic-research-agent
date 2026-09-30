# indic-research-agent

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Chainlit](https://img.shields.io/badge/UI-Chainlit-111827)](https://chainlit.io/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-0F766E)](https://langchain-ai.github.io/langgraph/)

`indic-research-agent` is an India- and Indic-language-focused research assistant. It searches public research providers, uses a LiteLLM-compatible model to synthesize evidence, and presents the answer through a Chainlit web UI.

The project is designed as a transparent research workflow: source search, tool calls, progress, citations, and uncertainty should remain visible instead of being hidden behind an unsupported retrieval claim.

## Table of contents

- [Problem statement](#problem-statement)
- [What the project does](#what-the-project-does)
- [Research workflow](#research-workflow)
- [Architecture](#architecture)
- [Running locally](#running-locally)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing and validation](#testing-and-validation)
- [Repository structure](#repository-structure)
- [Design constraints and limitations](#design-constraints-and-limitations)
- [Security notes](#security-notes)
- [Further documentation](#further-documentation)
- [Contributing](#contributing)
- [License](#license)

## Problem statement

Research questions about Indic languages and India-focused datasets, methods, and benchmarks often require several public sources and careful synthesis. This project provides a small agent workflow that can:

- search public research providers through `query-kit`;
- call a typed search tool from a LangGraph agent;
- synthesize source-backed answers with a LiteLLM-compatible model;
- show live progress and streamed output in Chainlit; and
- persist chat history, audit metadata, and cache entries for local development.

The current application is a public-provider research assistant. It does not claim to search a local document corpus when no ingestion or local retrieval pipeline is configured.

## What the project does

| Capability | Current behavior |
| --- | --- |
| Indic/India research | The system prompt specializes the agent for Indic-language and India-focused research questions. |
| Public research search | `query-kit` providers are called through `QueryKitService` and the typed `search` tool. |
| Answer synthesis | A LiteLLM/LangChain chat model is orchestrated through LangGraph. |
| Progress and streaming | Chainlit renders one live progress row and streams answer tokens when the selected model emits chunks. |
| Authentication | Local password authentication is provided by Chainlit through `AuthService`. |
| Chat history | Chainlit history is stored in PostgreSQL under the `chainlit` schema. |
| App audit data | Queries, tool calls, responses, and cache metadata are stored in the public application schema. |
| Caching | Redis caches normalized public-search/tool results. |
| Observability | Optional Phoenix/OpenTelemetry tracing records agent and provider spans. |

## Research workflow

The runtime path is intentionally narrow and testable:

1. A user asks a research question in the Chainlit UI.
2. The agent asks a clarification question only when language, domain, timeframe, provider coverage, or output format would materially change the answer.
3. LangGraph calls the typed public-research `search` tool when evidence is needed.
4. `QueryKitService` queries the configured public providers and normalizes their results.
5. The model synthesizes an answer with source identifiers and evidence boundaries.
6. Chainlit renders progress, streamed output, and the completed answer.
7. PostgreSQL and Redis retain the configured history, audit, and cache state.

`query-kit` is the project retrieval boundary for the current app. The project policy remains keyword/BM25-first and explicitly avoids embeddings and vector databases.

### Runtime data flow

```text
Browser
  -> Chainlit authentication and session
  -> Chainlit callbacks
  -> ChatController
  -> AgentService
  -> LangGraph agent
  -> LiteLLM-compatible model
  -> typed public-research search tool
  -> query-kit providers
  -> streamed answer and progress events
  -> PostgreSQL history/audit + Redis cache
```

## Architecture

### Compiled LangGraph

The agent graph is a two-node tool-calling loop compiled by
`build_agent_graph()`:

![Compiled LangGraph agent flow](docs/diagrams/langgraph-agent.png)

The transitions shown in the generated graph are:

- `__start__ -> llm`: begin with the LiteLLM-compatible model.
- `llm -> tools`: the model emitted a tool call and the tool-call budget is
  still available.
- `tools -> llm`: execute the public research search and return its results to
  the model.
- `llm -> __end__`: the model returned an answer without a tool call, or the
  `max_tool_calls` limit was reached.

The graph state carries `messages`, `tool_call_count`, `retrieved_context`, and
`final_answer`. The diagram is generated from the compiled graph rather than
hand-drawn. Regenerate it after changing `src/indic_research_agent/agent/graph.py`:

```bash
uv run python scripts/export_agent_graph.py
```

The exporter uses a compile-only model stub; it does not call an LLM or a
public research provider.

### Surrounding service boundaries

- Chainlit callbacks call `ChatController`, which delegates to
  `AgentService` for framework-neutral graph events and answer streaming.
- The `tools` node invokes `SearchTool`, which uses `QueryKitService` to query
  configured public research providers and normalize results.
- PostgreSQL stores Chainlit history and application audit rows; Redis stores
  search/tool cache entries outside the LangGraph state.

Chainlit remains a thin UI adapter. Orchestration belongs in services and LangGraph modules; retrieval and tool behavior should remain testable without a live LLM.

## Running locally

### Prerequisites

- Python `3.12`
- [`uv`](https://docs.astral.sh/uv/)
- Git
- Docker and Docker Compose for PostgreSQL, Redis, and the full application stack
- A configured LiteLLM/native provider key for live model answers

### 1. Clone and install

```bash
git clone git@github.com:neeraj1909/indic-research-agent.git
cd indic-research-agent

uv sync
cp .env.example .env
```

Edit `.env` before starting the application. At minimum, configure a usable model path and replace the development authentication secret for any non-local deployment.

### 2. Start the full stack with Docker

```bash
docker compose up --build
```

Then open [http://localhost:8000](http://localhost:8000).

The local development credentials are:

- username: `test`
- password: `test1234`

These values are convenience defaults only. Change them, and set a strong `CHAINLIT_AUTH_SECRET`, before exposing the application beyond local use.

### 3. Run Chainlit locally

Use this route when you want the application process outside Docker while PostgreSQL and Redis remain containerized:

```bash
docker compose up -d postgres redis
uv run alembic upgrade head
uv run chainlit run src/indic_research_agent/ui/chainlit_app.py -w
```

## Configuration

The complete configuration template is in [`.env.example`](.env.example). The most important settings are:

| Variable | Purpose |
| --- | --- |
| `LITELLM_MODEL` | Model identifier used for answer synthesis and tool-calling decisions. |
| `LITELLM_API_KEY` / provider keys | Credentials for the selected LiteLLM/native provider. |
| `LITELLM_API_BASE` | Optional OpenAI-compatible proxy or provider base URL. |
| `QUERY_KIT_PROVIDERS` | Comma-separated public research provider IDs. |
| `QUERY_KIT_TIMEOUT_SECONDS` | Overall public-provider search budget. |
| `APP_DATABASE_URL` | SQLAlchemy URL for app audit and cache metadata. |
| `REDIS_URL` | Redis URL for tool-result caching. |
| `CHAINLIT_DATABASE_URL` | PostgreSQL URL for Chainlit history. |
| `CHAINLIT_DATABASE_SCHEMA` | PostgreSQL schema for Chainlit tables; defaults to `chainlit`. |
| `CHAINLIT_AUTH_SECRET` | Secret used to sign Chainlit authentication cookies. |
| `PHOENIX_ENABLED` | Enables optional Phoenix/OpenTelemetry tracing. |

The default provider set is:

```text
semantic-scholar,semantic-scholar-web,pubmed,arxiv-web
```

Provider-specific availability, quotas, and credentials can vary. The app continues with partial public results when a provider times out or fails where the adapter can safely do so.

### Authorized browser cookie synchronization

The optional browser-cookie jar is for authorized automation against an already-running Chrome/Chromium DevTools Protocol endpoint. This repository does not install or launch a browser. Cookie values are secrets: do not commit, print, or share the jar.

## Usage

After the app starts:

1. Open [http://localhost:8000](http://localhost:8000).
2. Sign in with the configured local credentials.
3. Ask a research question.

Example prompts:

```text
Summarize recent research on Hindi OCR.
```

```text
Compare datasets for Indian-language ASR.
```

```text
Find sources on Marathi legal text classification.
```

```text
Create a research brief on Indic-language evaluation benchmarks.
```

During a run, the UI shows one progress status row for request receipt, model calls, search-tool activity, and final synthesis. The row is removed when the run succeeds or fails. If the model returns incremental chunks, answer text streams into the message; otherwise the completed answer is rendered at the end.

Optional Chainlit uploads may be accepted by the UI configuration, but this application version does not ingest uploads into a searchable local corpus.

### Deterministic smoke query

This path validates the agent/tool/cache/persistence wiring without live model credentials:

```bash
docker compose up -d postgres redis

APP_DATABASE_URL='postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent' \
REDIS_URL='redis://localhost:6379/0' \
uv run python scripts/smoke_query.py \
  --question "Find public research on Hindi OCR."
```

The smoke script uses deterministic stand-ins for the model and public provider, then checks search results, a Redis cache hit, and persisted PostgreSQL rows.

## Testing and validation

The project defines `unit`, `integration`, and `e2e` pytest markers.

### Formatting and unit tests

```bash
uv run python -m compileall src migrations scripts
uv run ruff format . && uv run ruff check --fix .
timeout 60 uv run pytest -m unit --no-cov
```

### Integration tests

Integration tests require PostgreSQL and Redis:

```bash
docker compose up -d postgres redis
uv run pytest -m integration
```

### End-to-end tests

Start the app first, then run:

```bash
uv run pytest -m e2e
```

### Provider adapter probe

This probe exercises the `query-kit` adapter without Docker, Chainlit, or an LLM:

```bash
uv run python scripts/probe_querykit_providers.py \
  --query "Hindi OCR" \
  --limit 3 \
  --since-year 2020
```

For changes affecting Chainlit, authentication, streaming, migrations, or provider adapters, run the relevant integration/e2e checks in addition to the unit suite.

## Repository structure

```text
.
├── .chainlit/                 # Chainlit UI/session configuration
├── .env.example               # Local configuration template
├── docs/
│   ├── architecture.md        # Focused architecture notes
│   ├── diagrams/
│   │   └── langgraph-agent.png # Generated compiled graph topology
│   └── technical-report.md    # Research-style technical report
├── migrations/                # Alembic environment and revisions
├── scripts/
│   ├── export_agent_graph.py   # Generate the LangGraph topology image
│   ├── migrate.sh             # Run database migrations
│   ├── probe_querykit_providers.py
│   ├── smoke_query.py         # Deterministic end-to-end smoke
│   └── start_app.sh           # Migrate, sync optional cookies, start Chainlit
├── src/indic_research_agent/
│   ├── agent/                 # LangGraph graph, prompt, and model adapters
│   ├── controllers/           # UI-facing controllers
│   ├── db/                    # Async SQLAlchemy engine/session helpers
│   ├── models/                # SQLAlchemy models
│   ├── repositories/          # Persistence repositories
│   ├── services/              # Orchestration and external-service adapters
│   ├── tools/                 # Typed research tools
│   └── ui/                    # Chainlit adapter and stream renderer
├── tests/
│   ├── unit/                  # Fast isolated tests
│   ├── integration/           # PostgreSQL/Redis/process tests
│   └── e2e/                   # Running-app smoke tests
├── compose.yaml
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Design constraints and limitations

- The initial retrieval policy is keyword/BM25-first. Do not add embeddings, vector databases, `pgvector`, FAISS, Chroma, Milvus, Weaviate, Pinecone, sentence-transformers, or embedding API calls without a new evidence-backed architecture decision.
- The default app searches public providers through `query-kit`; local/project document ingestion and retrieval are not implemented in this app version.
- Live answers require a configured model provider. Deterministic tests and smoke scripts can run without live model credentials.
- Chainlit text history and metadata are persisted in PostgreSQL. No object storage provider is configured for durable binary uploads.
- Chainlit realtime state is not configured for multi-replica Socket.IO scaling; keep the app deployment at one replica unless that architecture is revisited.
- `allow_origins = ["*"]` is a local-development setting and should be restricted before production exposure.

## Security notes

- Never commit `.env`, API keys, authentication secrets, or browser cookie jars.
- Replace `CHAINLIT_AUTH_SECRET`, `CHAINLIT_AUTH_USERNAME`, and `CHAINLIT_AUTH_PASSWORD` outside local/e2e use.
- Treat chat messages, thread metadata, tool arguments, answers, and audit rows as potentially sensitive research data.
- Use `BROWSER_CDP_ENDPOINT` only for a browser session you are authorized to control.
- Review provider terms, quotas, and data-handling requirements before sending sensitive research questions to external services.
- The repository currently has no `LICENSE` file; see [License](#license).

## Further documentation

- [Architecture notes](docs/architecture.md)
- [Technical report](docs/technical-report.md)
- [Environment template](.env.example)
- [Agent instructions](AGENTS.md)

## Contributing

1. Create a focused topic branch.
2. Keep Chainlit-specific code in the UI adapter; put orchestration in services and agent modules.
3. Keep provider/search behavior testable without an LLM.
4. Add or update unit tests for behavior changes and integration/e2e coverage when external services or runtime boundaries change.
5. Add an Alembic migration for database schema changes.
6. Run the relevant validation commands before opening a pull request.
7. Do not introduce vector/embedding dependencies or local retrieval behavior without updating the project policy, tests, and documentation.

## License

No license file is currently included. Treat the repository as unlicensed unless the project owner adds a license or documents licensing terms.
