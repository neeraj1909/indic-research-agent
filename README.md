# indic-research-agent

`indic-research-agent` is a BM25-first research assistant with a Chainlit web UI,
LangGraph tool orchestration, LiteLLM model access, query-kit public research
search, PostgreSQL persistence, and Redis-backed caching.

The project is intentionally keyword/BM25-first. It does **not** use embeddings,
vector databases, `pgvector`, FAISS, Chroma, Milvus, Weaviate, Pinecone,
sentence-transformers, or embedding API calls. A unit test guards against adding
common vector/embedding packages as direct dependencies.

## Table of contents

- [Project overview](#project-overview)
- [Architecture](#architecture)
- [Repository structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the application](#running-the-application)
- [Usage guide](#usage-guide)
- [Testing and validation](#testing-and-validation)
- [Development guide](#development-guide)
- [Deployment notes](#deployment-notes)
- [Troubleshooting](#troubleshooting)
- [Security notes](#security-notes)
- [Contributing](#contributing)
- [License](#license)

## Project overview

### What it does

The application answers research questions by combining:

1. local keyword retrieval over document chunks using BM25;
2. public research search through `query-kit` providers;
3. a LangGraph tool-calling loop with `search` and `fetch` tools;
4. a LiteLLM-compatible chat model for synthesis;
5. a Chainlit UI with local password auth, resumable chat history, progress
   steps, and answer streaming/fallback updates.

### Core use cases

- Explore a small local document corpus without vector infrastructure.
- Search public research providers through query-kit from the same agent flow.
- Evaluate BM25-first retrieval and tool-calling behavior with deterministic
  tests and smoke scripts.
- Run a local Chainlit chat surface that shows progress while long research
  calls execute.
- Persist Chainlit chat history separately from app audit tables.

### High-level capabilities

| Capability | Current implementation |
| --- | --- |
| Web UI | Chainlit app in `src/indic_research_agent/ui/chainlit_app.py`. |
| Authentication | Chainlit password auth via `AuthService`; local/e2e user defaults to `test` / `test1234`. |
| Chat history | Chainlit SQLAlchemy data layer backed by PostgreSQL schema `chainlit`; `@cl.on_chat_resume` restores runtime state. |
| Streaming/progress | Agent emits framework-neutral events; Chainlit renders progress `Step`s and streams answer text when chunks are available. |
| Retrieval | In-memory BM25 index over seed chunks for the default UI graph; database-backed chunks are available through services and smoke tests. |
| Public research search | `QueryKitService` wraps `query-kit` providers. |
| Persistence/audit | App tables store users, documents, chunks, queries, tool calls, responses, and cache metadata. |
| Cache | Redis JSON cache for query-kit/tool results, with namespace TTL policy. |
| Model provider | LiteLLM via `langchain-litellm`; optional OpenAI-compatible streaming proxy adapter for `chatgpt/*` model IDs. |
| Docker | Compose starts app, PostgreSQL, and Redis; the app runs Alembic migrations before Chainlit. |

## Architecture

### Main components

| Layer | Files | Responsibility |
| --- | --- | --- |
| Chainlit UI adapter | `src/indic_research_agent/ui/chainlit_app.py`, `src/indic_research_agent/ui/chainlit_stream_renderer.py` | Defines Chainlit callbacks, password auth callback, data-layer callback, session setup/resume, and rendering of service events as messages/steps. |
| Controller | `src/indic_research_agent/controllers/chat_controller.py` | Thin UI-facing boundary. Holds JSON-safe message history/session metadata and delegates to `AgentService`. |
| Agent service | `src/indic_research_agent/services/agent_service.py`, `src/indic_research_agent/services/agent_events.py` | Converts chat history into LangChain messages, streams LangGraph events, preserves the one-shot `answer()` API, and optionally records query/tool/response audit rows. |
| LangGraph agent | `src/indic_research_agent/agent/graph.py`, `state.py`, `prompts.py`, `llm.py`, `openai_streaming.py` | Builds the LLM/tool loop, binds tools, emits custom progress events, and creates the LiteLLM-backed model. |
| Tools | `src/indic_research_agent/tools/search.py`, `fetch.py`, `schemas.py` | Typed `search` and `fetch` tool implementations. |
| Retrieval | `src/indic_research_agent/retrieval/` | BM25 tokenizer/index, search/fetch contracts, and seed corpus. |
| External research | `src/indic_research_agent/services/querykit_service.py` | Async wrapper around `query-kit` search providers with fallback behavior. |
| Persistence | `src/indic_research_agent/models/`, `repositories/`, `services/query_service.py`, `services/document_service.py` | SQLAlchemy models/repositories and service APIs for app-owned audit/document/cache data. |
| Chainlit persistence | `src/indic_research_agent/services/chainlit_data_layer.py`, `migrations/versions/20260505_0002_chainlit_schema.py`, `20260505_0003_chainlit_step_autocollapse.py` | Chainlit SQLAlchemy data layer configured with Postgres `search_path=chainlit` and its required tables. |
| Cache | `src/indic_research_agent/services/cache_service.py`, `cache_policy.py` | Redis-backed JSON cache and namespace TTL policy. |

### Runtime data flow

```text
Browser
  -> Chainlit authenticated session
  -> Chainlit callbacks in ui/chainlit_app.py
  -> ChatController
  -> AgentService.stream_answer(...)
  -> LangGraph graph.astream(stream_mode=["updates", "messages", "custom"])
  -> LiteLLM chat model
  -> search/fetch tools
  -> local BM25 SearchService and/or query-kit public providers
  -> AgentStreamEvent objects
  -> Chainlit Message.stream_token(...) and Step progress UI
  -> Chainlit history tables in schema chainlit
  -> optional app audit rows in public queries/tool_calls/agent_responses
```

### Agent workflow

The system prompt in `src/indic_research_agent/agent/prompts.py` makes the
assistant an Indic-language and India-focused research assistant. It instructs
the agent to:

- specialize in Indic/India research tasks such as Hindi OCR, Indian-language
  ASR, Marathi legal text classification, Tamil passage translation/analysis,
  and Indic evaluation benchmarks;
- ask clarification questions only when language, script, domain, timeframe,
  corpus, or output format would materially change the answer;
- use `search` before factual, recent, comparative, dataset, benchmark,
  source-finding, or literature-review answers;
- search both local BM25 and query-kit public providers by default with
  `source="all"`;
- use `fetch` only when a local BM25 search result needs more detail, not for
  query-kit result IDs;
- cite source identifiers and explain evidence boundaries/uncertainty;
- not claim semantic/vector retrieval, embeddings, or unsupported app features
  were used.

`build_agent_graph()` prepends this project prompt to every model call and
strips incoming system messages so UI/session history cannot silently replace it.
Runtime logs include only the prompt version/hash, not the prompt text.

The graph in `src/indic_research_agent/agent/graph.py` has two nodes:

1. `llm`: calls the bound LiteLLM/LangChain chat model.
2. `tools`: executes requested `search` or `fetch` tool calls.

The loop stops when there are no tool calls or `max_tool_calls` is reached
(default: `6`).

Important current behavior:

- `search` can use `source="local"`, `source="research"`, or `source="all"`.
- `fetch` only fetches local document IDs/chunks from `SearchService`; it is not
  for query-kit result IDs.
- The default Chainlit agent currently uses `create_seed_search_service()`, so
  its local BM25 index is the built-in seed corpus. Database-backed document
  ingestion exists in `DocumentService` and is exercised by smoke/integration
  tests, but loading persisted documents into the default Chainlit graph is not
  currently wired as a runtime feature.

### Chainlit integration

`src/indic_research_agent/ui/chainlit_app.py` defines:

- `@cl.data_layer`: returns a cached `SQLAlchemyDataLayer` from
  `get_chainlit_data_layer()`.
- `@cl.password_auth_callback`: verifies local credentials via `AuthService` and
  returns a Chainlit `User` on success.
- `@cl.on_chat_start`: initializes JSON-safe session state.
- `@cl.on_chat_resume`: rehydrates controller/session state from Chainlit thread
  metadata or persisted message steps.
- `@cl.on_message`: creates a response message immediately, consumes
  `ChatController.stream_message(...)`, and delegates rendering to
  `ChainlitStreamRenderer`.

Chainlit UI configuration lives in `.chainlit/config.toml`.

### Streaming behavior

Streaming is split into two layers:

1. **Service layer:** `AgentService.stream_answer(...)` yields typed,
   framework-neutral events from `src/indic_research_agent/services/agent_events.py`:
   - `AgentRunStarted`
   - `AgentProgress`
   - `AgentToolStarted`
   - `AgentToolFinished`
   - `AgentToken`
   - `AgentCompleted`
   - `AgentFailed`

2. **UI layer:** `ChainlitStreamRenderer` converts those events into Chainlit
   `Message` and `Step` updates.

The UI always shows progress for request receipt, LLM calls, tool start/end,
final synthesis, completion, and errors. Answer text is streamed when the
selected LangGraph/model path yields message chunks; otherwise Chainlit still
shows progress and updates the final message from `AgentCompleted.answer`.

### Persistence and session handling

There are two PostgreSQL persistence namespaces:

| Namespace | Tables | Purpose |
| --- | --- | --- |
| public app schema | `users`, `documents`, `document_chunks`, `queries`, `tool_calls`, `agent_responses`, `cache_metadata` | App-owned domain/audit/cache metadata. |
| `chainlit` schema | `users`, `threads`, `steps`, `elements`, `feedbacks` | Chainlit data layer for authenticated users, chat history, steps, thread metadata, and feedback. |

Why separate schemas? Chainlit requires a `users` table with a different shape
from the app's public `users` table. The Chainlit data layer sets Postgres
`search_path` to `CHAINLIT_DATABASE_SCHEMA` (default: `chainlit`) to avoid table
collisions.

Runtime session details:

- Chainlit persists thread metadata on websocket disconnect.
- The app stores only JSON-safe session values such as `message_history`,
  `app_session_id`, `chainlit_thread_id`, and `user_identifier`.
- Runtime objects (`ChatController`, graph instances, database sessions, model
  clients) are recreated on chat start/resume/message.
- UI requests can also record app audit rows with the Chainlit thread/session ID
  in `queries.session_id`.

Binary elements/uploads: the SQLAlchemy data layer is configured without an S3,
Azure, or GCS storage provider. Text messages, steps, thread metadata, and
history persist; uploaded/binary Chainlit elements are not durable. The current
Chainlit config restricts spontaneous uploads to text, Markdown, PDF, JSON, and
CSV with small local-development limits.

### Authentication flow

Local/e2e auth uses `AuthService` in
`src/indic_research_agent/services/auth_service.py`:

1. Chainlit serves `/login` when auth is enabled.
2. The password callback compares submitted credentials with
   `CHAINLIT_AUTH_USERNAME` and `CHAINLIT_AUTH_PASSWORD` using timing-safe
   comparison.
3. On success, Chainlit creates a signed auth cookie using
   `CHAINLIT_AUTH_SECRET`.
4. On login, Chainlit creates/updates the persisted Chainlit user row in the
   `chainlit.users` table.

Default local/e2e credentials are `test` / `test1234`. They are not production
credentials.

### External services and providers

| Service/provider | Used for | Required for |
| --- | --- | --- |
| PostgreSQL | App tables and Chainlit history. | Integration tests, e2e smoke, Docker runtime, resumable Chainlit history. |
| Redis | JSON cache for query-kit/tool results. | Cache tests and smoke query cache validation. Not used for Chainlit Socket.IO scaling. |
| LiteLLM / OpenAI-compatible model | Final answer synthesis and tool-calling decisions. | Live Chainlit answers. Deterministic tests can run without live model credentials. |
| query-kit | Public research provider search. | Research-source search from the default agent. Provider-specific credentials are not documented in this repo. |

## Repository structure

```text
.
├── .chainlit/config.toml             # Chainlit runtime/UI config
├── .env.example                      # Example environment variables
├── AGENTS.md                         # Coding-agent instructions and active PRP pointer
├── alembic.ini                       # Alembic configuration
├── compose.yaml                      # Docker Compose app + Postgres + Redis
├── Dockerfile                        # App image build
├── docs/architecture.md              # Short architecture map
├── migrations/                       # Alembic environment and revisions
├── scripts/
│   ├── migrate.sh                    # Runs Alembic upgrade head
│   ├── start_app.sh                  # Runs migrations then Chainlit
│   └── smoke_query.py                # Deterministic end-to-end BM25/tool/cache smoke
├── src/indic_research_agent/
│   ├── agent/                        # LangGraph graph, prompt, model factory/adapters
│   ├── controllers/                  # UI-facing controller boundary
│   ├── db/                           # Async SQLAlchemy engine/session helpers
│   ├── models/                       # SQLAlchemy ORM models
│   ├── repositories/                 # Persistence repository classes
│   ├── retrieval/                    # BM25 index and search contracts
│   ├── services/                     # App orchestration and external service adapters
│   ├── tools/                        # Search/fetch tools and schemas
│   └── ui/                           # Chainlit adapter and stream renderer
└── tests/
    ├── e2e/                          # Live/smoke end-to-end tests
    ├── integration/                  # Postgres/Redis integration tests
    └── unit/                         # Fast unit tests
```

## Prerequisites

### Required

- Python `3.12` (`pyproject.toml` requires `>=3.12,<3.13`; `.python-version`
  contains `3.12`).
- [`uv`](https://docs.astral.sh/uv/) for dependency management and command
  execution.
- `git`, because the project depends on `query-kit` from
  `git+https://github.com/neeraj1909/query-kit.git`.

### Required for full runtime/integration tests

- PostgreSQL reachable with an asyncpg SQLAlchemy URL.
- Redis reachable with a Redis URL.
- Docker/Compose if using the provided container stack.

### Required for live model answers

At least one model provider path must be configured. The code supports:

- LiteLLM/LangChain via `langchain-litellm` settings such as `LITELLM_MODEL`,
  `LITELLM_API_KEY`, and `LITELLM_API_BASE`.
- Native provider env vars such as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`,
  depending on the chosen LiteLLM model/provider.
- An OpenAI-compatible streaming proxy path for `chatgpt/*` model IDs when
  `LITELLM_STREAMING=true`, `LITELLM_API_BASE` is set, and an API key is set.

## Installation

### 1. Clone and enter the repository

```bash
git clone <repository-url>
cd indic-research-agent
```

Replace `<repository-url>` with the actual repository URL you use. The README
cannot verify a canonical clone URL from the source tree alone.

### 2. Install dependencies

```bash
uv sync
```

This creates/updates `.venv/` and installs runtime and dev dependencies from
`pyproject.toml` and `uv.lock`.

### 3. Create local environment file

```bash
cp .env.example .env
```

Edit `.env` with your model provider values and, for non-local use, replace the
Chainlit auth secret/credentials.

### 4. Verify imports

```bash
uv run python -c "import indic_research_agent; import query_cli"
```

## Configuration

Configuration is loaded by `AppSettings` in `src/indic_research_agent/config.py`.
It reads environment variables and `.env` with `extra="ignore"`.

### Environment variables

| Variable | Default / example | Purpose |
| --- | --- | --- |
| `APP_NAME` | `indic-research-agent` | Application name. |
| `ENVIRONMENT` | `development` | Environment label. |
| `LOG_LEVEL` | `INFO` | Intended log level setting. Logging configuration is otherwise framework/default-driven. |
| `APP_DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent` | App SQLAlchemy URL. Preferred over `DATABASE_URL` for app tables. |
| `DATABASE_URL` | none | Accepted as fallback alias for app DB URL. Avoid relying on it for Chainlit; use `CHAINLIT_DATABASE_URL` explicitly. |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis cache URL. |
| `QUERY_KIT_PROVIDERS` | `semantic-scholar,semantic-scholar-web,pubmed,arxiv-web` | Comma-separated provider IDs. Supported values include `acl`, `arxiv`, `arxiv-web`, `pubmed`, `semantic-scholar`, `semantic-scholar-web`, `openreview`, and `all`. Default uses the measured fast/partial-safe provider set plus safe public-web fallbacks for Indic OCR queries. |
| `QUERY_KIT_TIMEOUT_SECONDS` | `30` | Overall public-provider budget for query-kit search; the UI emits per-provider progress/timeout steps and continues with partial/local BM25 results on timeout. |
| `QUERY_CLI_USER_AGENT` | `indic-research-agent/0.1 (+https://github.com/neeraj1909/indic-research-agent)` | Project-specific User-Agent passed through query-kit for ordinary HTTP public-web providers such as `arxiv-web` and `semantic-scholar-web`. |
| `BROWSER_COOKIE_JAR_ENABLED` | `true` | Enables the persistent browser-cookie jar service for authorized browser automation. When enabled without `BROWSER_CDP_ENDPOINT`, the jar can still be created but CDP sync is a no-op. |
| `BROWSER_CDP_ENDPOINT` | none | Optional authorized Chrome/Chromium DevTools Protocol endpoint (`http://host:9222` or `ws://.../devtools/browser/...`). The app does not install or launch Chrome. |
| `CDP_ENDPOINT` | none | Backward-compatible alias for `BROWSER_CDP_ENDPOINT`. |
| `BROWSER_COOKIE_JAR_PATH` | `tmp/browser-cookie-jar/cookies.local.json` in code; Compose defaults to `/data/browser-cookies/cookies.json` | Disk JSON cookie jar path. Compose mounts `/data/browser-cookies` from the `browser_cookie_data` named volume so the jar survives app container recreation. |
| `BROWSER_COOKIE_JAR_SYNC_INTERVAL_SECONDS` | `30` | Background CDP-to-disk cookie sync interval. |
| `BROWSER_COOKIE_JAR_STARTUP_TIMEOUT_SECONDS` | `10` | Startup CDP apply/read timeout budget. |
| `BROWSER_COOKIE_JAR_FAIL_ON_ERROR` | `false` | If true, cookie-sync failures exit non-zero; by default they are logged/surfaced as sanitized status and do not block the app. |
| `PHOENIX_ENABLED` | `false` in code; Compose/.env example use `true` | Enables OpenTelemetry trace export to Phoenix. |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://10.20.30.1:16006` | Phoenix app hostname; OTLP/HTTP traces are sent to `/v1/traces` under this endpoint. |
| `PHOENIX_PROJECT_NAME` | `indic-research-agent` | Phoenix project name for agent traces. |
| `PHOENIX_PROTOCOL` | `http/protobuf` | OTLP transport protocol used by the app. |
| `PHOENIX_BATCH_SPANS` | `true` | Batch spans before export; app force-flushes at end of each agent run. |
| `PHOENIX_AUTO_INSTRUMENT` | `true` | Activates installed OpenInference LangChain/LiteLLM instrumentors in addition to app manual spans. |
| `LITELLM_MODEL` | code default `openai/gpt-4o-mini`; `.env.example` uses `chatgpt/gpt-5.5` | Model ID for LiteLLM/LangChain. |
| `LITELLM_CUSTOM_LLM_PROVIDER` | none | Optional LiteLLM custom provider name. |
| `LITELLM_API_KEY` | none | Optional API key passed to LiteLLM or the OpenAI-compatible adapter. |
| `LITELLM_API_BASE` | none | Optional API base/proxy URL. |
| `LITELLM_TEMPERATURE` | `0.2` | Model temperature. |
| `LITELLM_TIMEOUT_SECONDS` | `30` | Model request timeout. |
| `LITELLM_MAX_TOKENS` | `1200` in code; Compose/.env example use `25600` | Maximum model output tokens when supported. |
| `LITELLM_STREAMING` | `true` | Enables streaming-oriented model configuration. |
| `OPENAI_API_KEY` | none | Native provider env var that LiteLLM may use. |
| `ANTHROPIC_API_KEY` | none | Native provider env var that LiteLLM may use. |
| `CHAINLIT_AUTH_SECRET` | no code default; Compose/.env example use a dev-only value | Required by Chainlit to sign auth cookies when auth is enabled. Replace in non-local environments. |
| `CHAINLIT_AUTH_ENABLED` | `true` | Enables local password credential verification. |
| `CHAINLIT_AUTH_USERNAME` | `test` | Local/e2e username. |
| `CHAINLIT_AUTH_PASSWORD` | `test1234` | Local/e2e password. |
| `CHAINLIT_DATABASE_URL` | same local Postgres URL by default | SQLAlchemy URL for Chainlit's data layer. |
| `CHAINLIT_DATABASE_SCHEMA` | `chainlit` | Postgres schema used for Chainlit tables/search path. |
| `CHAINLIT_DATA_LAYER_SHOW_LOGGER` | `false` | Enables Chainlit SQLAlchemy data-layer logging if set truthy. |
| `COMPOSE_PROJECT_NAME` | `indic-research-agent` in `.env.example` | Compose project name. |

### Example local `.env`

```dotenv
APP_NAME=indic-research-agent
ENVIRONMENT=development
LOG_LEVEL=INFO

# Model provider. Fill these in for live UI answers.
LITELLM_MODEL=openai/gpt-4o-mini
LITELLM_CUSTOM_LLM_PROVIDER=
LITELLM_API_KEY=
LITELLM_API_BASE=
LITELLM_TEMPERATURE=0.2
LITELLM_TIMEOUT_SECONDS=30
LITELLM_MAX_TOKENS=25600
LITELLM_STREAMING=true
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Research providers.
QUERY_KIT_PROVIDERS=semantic-scholar,semantic-scholar-web,pubmed,arxiv-web
QUERY_KIT_TIMEOUT_SECONDS=30
QUERY_CLI_USER_AGENT=indic-research-agent/0.1 (+https://github.com/neeraj1909/indic-research-agent)

# Authorized browser cookie persistence via external CDP.
BROWSER_COOKIE_JAR_ENABLED=true
BROWSER_CDP_ENDPOINT=
# Local default if omitted: tmp/browser-cookie-jar/cookies.local.json
# Docker Compose defaults this to /data/browser-cookies/cookies.json.
# BROWSER_COOKIE_JAR_PATH=tmp/browser-cookie-jar/cookies.local.json
BROWSER_COOKIE_JAR_SYNC_INTERVAL_SECONDS=30
BROWSER_COOKIE_JAR_STARTUP_TIMEOUT_SECONDS=10
BROWSER_COOKIE_JAR_FAIL_ON_ERROR=false

# Phoenix observability.
PHOENIX_ENABLED=true
PHOENIX_COLLECTOR_ENDPOINT=http://10.20.30.1:16006
PHOENIX_PROJECT_NAME=indic-research-agent
PHOENIX_PROTOCOL=http/protobuf
PHOENIX_BATCH_SPANS=true
PHOENIX_AUTO_INSTRUMENT=true

# Local services.
APP_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent
REDIS_URL=redis://localhost:6379/0

# Chainlit local/e2e auth and history.
CHAINLIT_AUTH_SECRET=replace-with-a-long-random-secret
CHAINLIT_AUTH_ENABLED=true
CHAINLIT_AUTH_USERNAME=test
CHAINLIT_AUTH_PASSWORD=test1234
CHAINLIT_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent
CHAINLIT_DATABASE_SCHEMA=chainlit
```

### Persistent browser cookie jar

The browser cookie jar is for **authorized browser automation sessions only**.
It talks to an already-running Chrome/Chromium DevTools Protocol endpoint with
`Storage.setCookies` and `Storage.getCookies`; this repository does not install,
launch, or manage Chrome.

Operational behavior:

1. `CookieJarStore` creates a versioned JSON jar if it is missing and writes it
   atomically with restrictive permissions where the filesystem supports chmod.
2. `scripts/sync_browser_cookies.py --startup` applies existing jar cookies to
   the configured CDP endpoint and then snapshots active browser cookies back to
   disk.
3. `scripts/sync_browser_cookies.py --watch` keeps syncing at
   `BROWSER_COOKIE_JAR_SYNC_INTERVAL_SECONDS` and performs a final sync on
   graceful shutdown.
4. `scripts/start_app.sh` runs the watcher as a background child before
   Chainlit when `BROWSER_COOKIE_JAR_ENABLED` is truthy.

Docker Compose mounts `browser_cookie_data:/data/browser-cookies`, so the
container jar path defaults to `/data/browser-cookies/cookies.json` and survives
app container recreation. Local non-Docker runs can omit `BROWSER_COOKIE_JAR_PATH`
to use the git-ignored default `tmp/browser-cookie-jar/cookies.local.json`.

Security boundaries:

- Cookie values are secrets. Do not commit, paste, print, trace, or share cookie
  jar files. `tmp/` and `*.local.json` are ignored for local probes.
- Logs and CLI status intentionally report only sanitized fields such as action,
  counts, endpoint-configured yes/no, and jar path; cookie names/values are not
  emitted.
- Set `BROWSER_CDP_ENDPOINT` only to an active browser you are authorized to use.
  Do not use this feature to bypass access controls, paywalls, rate limits, WAFs,
  or API-key requirements.
- The current deployment assumes one app writer for the jar. Multi-replica cookie
  jar writes are unsupported until a separate coordination design exists.

Manual sanitized probe:

```bash
BROWSER_CDP_ENDPOINT=http://127.0.0.1:9222 \
BROWSER_COOKIE_JAR_PATH=tmp/browser-cookie-jar/live-cookies.local.json \
uv run python scripts/sync_browser_cookies.py --once --json | jq '{ok,action,cookie_count,jar_path}'
```

### Chainlit settings

`.chainlit/config.toml` controls UI/session behavior. Notable current settings:

- `session_timeout = 3600`
- `user_session_timeout = 1296000` (15 days)
- `persist_user_env = false`
- `mask_user_env = false`
- `allow_origins = ["*"]` (review before production)
- `features.spontaneous_file_upload.enabled = true`
- upload accept list: text, Markdown, PDF, JSON, CSV
- `UI.name = "Indic Research Agent"`
- `UI.cot = "full"`, so progress/tool steps are visible

## Running the application

### Option A: Docker Compose (recommended for the full stack)

```bash
cp .env.example .env
# Edit .env with model provider values and a unique CHAINLIT_AUTH_SECRET.
docker compose up --build
```

The app container runs:

```bash
sh scripts/migrate.sh
uv run --no-sync chainlit run src/indic_research_agent/ui/chainlit_app.py --headless --host 0.0.0.0 --port 8000
```

Expected services:

```bash
docker compose ps
```

- `app` listening on `http://localhost:8000`
- `postgres` healthy on localhost port `5432`
- `redis` healthy on localhost port `6379`

Compose mounts `browser_cookie_data:/data/browser-cookies` for the persistent browser cookie jar. The jar path defaults to `/data/browser-cookies/cookies.json` inside the app container and survives app container recreation. Set `BROWSER_CDP_ENDPOINT` only to an authorized active browser CDP endpoint; this app does not install or launch Chrome/Chromium.

### Option B: Local Chainlit process with separately running services

Start PostgreSQL and Redis first. One convenient way is:

```bash
docker compose up -d postgres redis
```

Then run migrations and Chainlit locally:

```bash
cp .env.example .env
# Edit .env.
uv sync
uv run alembic upgrade head
uv run chainlit run src/indic_research_agent/ui/chainlit_app.py -w
```

Open:

```text
http://localhost:8000
```

With auth enabled, Chainlit redirects to `/login`.

### Alternative entry points

Dry-run model adapter smoke (does not call a live provider):

```bash
uv run python -m indic_research_agent.agent.llm --smoke
```

Live model smoke:

```bash
uv run python -m indic_research_agent.agent.llm --smoke --live --prompt "Reply with OK."
```

Deterministic end-to-end smoke query without live model credentials:

```bash
docker compose up -d postgres redis
APP_DATABASE_URL='postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent' \
REDIS_URL='redis://localhost:6379/0' \
uv run alembic upgrade head
APP_DATABASE_URL='postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent' \
REDIS_URL='redis://localhost:6379/0' \
uv run python scripts/smoke_query.py --question "How does BM25 retrieval support this agent?"
```

Expected smoke output is JSON with a non-empty `answer`, a `tool_call_count` of
`2`, at least `2` persisted tool calls, and at least `2` Redis cache hits.

## Usage guide

### Login

1. Open `http://localhost:8000`.
2. Sign in with the configured local/e2e credentials.
   - Defaults: `test` / `test1234`.
   - Chainlit may label the username field as an email field, but the app passes
     the value to `CHAINLIT_AUTH_USERNAME`.

### Asking questions

Example prompts:

```text
Summarize recent research on Hindi OCR.
```

```text
Compare datasets for Indian language ASR.
```

```text
Find sources on Marathi legal text classification.
```

```text
Translate and analyze this Tamil passage: ...
```

```text
Create a research brief on Indic language evaluation benchmarks.
```

The UI creates a response immediately, then shows progress steps such as:

- request received;
- LLM call;
- search/fetch tool start/end;
- final synthesis;
- answer finalized;
- completed or failed.

If the model/tool path emits answer chunks, the response streams into the
message. If not, the final answer is still displayed once the run completes.

### Chat history and resume

With auth and the Chainlit data layer enabled:

- prior threads are visible in Chainlit history;
- a thread can be resumed after reload or container restart when using the same
  Postgres volume/data;
- model context is restored from JSON-safe `message_history` metadata when
  available, or reconstructed best-effort from persisted user/assistant message
  steps.

### Inputs and outputs

- Input: normal text chat messages.
- Optional uploads are allowed by Chainlit config for text, Markdown, PDF, JSON,
  and CSV. No custom upload-processing pipeline is currently implemented in the
  agent code.
- Output: Chainlit assistant message plus progress/tool steps. App audit rows
  are written when UI persistence is enabled.

## Testing and validation

Pytest markers are configured in `pyproject.toml`:

- `unit`: fast unit tests;
- `integration`: tests requiring external services or process-level integration;
- `e2e`: end-to-end application smoke tests.

### Static checks

```bash
uv run python -m compileall src migrations scripts
uv run ruff format .
uv run ruff check --fix .
```

No separate type checker is configured in this repository.

### Unit tests

```bash
timeout 60 uv run pytest -m unit --no-cov
```

Unit coverage includes BM25 behavior, tool schemas/tools, query-kit wrapper
fallbacks, auth, chat history, controller delegation, stream event mapping,
LangGraph streaming, Chainlit callback helpers, and the no-vector-dependency
policy.

### Integration tests

Requires PostgreSQL and Redis for the full integration suite:

```bash
docker compose up -d postgres redis
APP_DATABASE_URL='postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent' \
REDIS_URL='redis://localhost:6379/0' \
uv run pytest -m integration
```

Integration tests cover:

- database-backed BM25 roundtrip;
- Redis cache behavior;
- repository roundtrip;
- Chainlit schema migration isolation;
- UI-style query/tool/response audit persistence.

### E2E tests

With the app running at `http://localhost:8000`:

```bash
APP_DATABASE_URL='postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent' \
REDIS_URL='redis://localhost:6379/0' \
uv run pytest -m e2e
```

`tests/e2e/test_chainlit_auth_persistence.py` also respects:

```bash
CHAINLIT_BASE_URL=http://localhost:8000
```

E2E coverage includes:

- `/auth/config` showing password auth enabled;
- `/login` with `test` / `test1234`;
- `/user` returning identifier `test`;
- `/project/settings?language=en-US` showing `dataPersistence=true` and
  `threadResumable=true`;
- deterministic smoke query through the BM25/tool/cache/persistence stack;
- service-level streaming smoke.

### Full local validation loop

```bash
uv run python -m compileall src migrations scripts
uv run ruff format . && uv run ruff check --fix .
timeout 60 uv run pytest -m unit --no-cov
docker compose up -d postgres redis
APP_DATABASE_URL='postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent' \
REDIS_URL='redis://localhost:6379/0' \
uv run pytest -m integration
docker compose up -d --build app
APP_DATABASE_URL='postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent' \
REDIS_URL='redis://localhost:6379/0' \
uv run pytest -m e2e
git diff --check
```

## Development guide

### Coding conventions

- Python target: 3.12.
- Formatting/linting: Ruff (`line-length = 88`, double quotes, import sorting,
  selected lint rules `E`, `F`, `I`, `UP`, `B`, `SIM`).
- Keep Chainlit-specific imports in UI/adapter modules where possible.
- Keep retrieval logic testable without an LLM.
- Maintain BM25/keyword-first retrieval unless a future plan records evidence to
  add vector/embedding infrastructure.

### Adding or modifying tools

1. Add or update input/output schemas in `src/indic_research_agent/tools/schemas.py`.
2. Implement the tool class in `src/indic_research_agent/tools/`.
3. Register it in `build_agent_graph()` / `_create_langchain_tools()` in
   `src/indic_research_agent/agent/graph.py`.
4. Emit custom progress with `_emit_custom(...)` if the UI should display
   tool-specific progress.
5. Add unit tests under `tests/unit/tools/` and graph tests under
   `tests/unit/agent/`.
6. If tool results should be audited, update the `AgentToolFinished` metadata
   mapping in `AgentService`/graph helpers and integration tests.

### Changing prompts or agent behavior

- Main system prompt, prompt version, and non-secret prompt fingerprint:
  `src/indic_research_agent/agent/prompts.py`.
- LangGraph prompt injection/routing/tool loop:
  `src/indic_research_agent/agent/graph.py`.
- Model factory and provider behavior: `src/indic_research_agent/agent/llm.py`.
- OpenAI-compatible proxy adapter: `src/indic_research_agent/agent/openai_streaming.py`.

After changing behavior, run at least:

```bash
uv run pytest tests/unit/agent tests/unit/services/test_agent_service_streaming.py -q
```

Prompt changes should keep `tests/unit/agent/test_prompts.py` green so the
Indic-specific prompt is still loaded and cannot be overridden by an incoming
system message.

### Changing Chainlit UI behavior

- Chainlit lifecycle/auth/data-layer callbacks: `src/indic_research_agent/ui/chainlit_app.py`.
- Stream rendering: `src/indic_research_agent/ui/chainlit_stream_renderer.py`.
- Chainlit runtime config: `.chainlit/config.toml`.

Keep UI rendering separate from graph orchestration. The UI should consume
`AgentStreamEvent` objects rather than parsing raw LangGraph chunks.

### Adding persistence/auth features

- App domain models live in `src/indic_research_agent/models/` and public
  Alembic migrations.
- App repositories live in `src/indic_research_agent/repositories/`.
- App persistence services live in `src/indic_research_agent/services/`.
- Chainlit's data-layer schema lives in the `chainlit` Postgres schema; do not
  reuse the public `users` table for Chainlit auth/history.
- Add Alembic migrations for database changes and integration tests that verify
  the new schema.

### Logging and debugging tips

- LangGraph logs progress with names like `agent.llm.start`, `agent.llm.end`,
  `agent.tool.start`, and `agent.tool.end`.
- Chainlit UI logs message lifecycle events from `chainlit_app.py`.
- App persistence errors in the stream path are logged and do not crash the UI
  response by default.
- Docker logs:

```bash
docker compose logs -f app
```

- Verify rendered Chainlit auth/settings without browser selectors:

```bash
curl -fsS http://localhost:8000/auth/config | jq .
curl -fsS http://localhost:8000/project/settings?language=en-US | jq '{dataPersistence,threadResumable}'
```

## Deployment notes

### Docker image

The `Dockerfile` uses `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`, installs
`git`, copies `.chainlit`, `migrations`, `scripts`, and `src`, runs
`uv sync --frozen --no-dev`, and starts with `scripts/start_app.sh`.

### Database migrations

The app runs `uv run --no-sync alembic upgrade head` before starting Chainlit in
Docker. Fresh Compose startup creates:

- public app tables from `20260505_0001_initial_schema.py`;
- Chainlit schema/tables from `20260505_0002_chainlit_schema.py`;
- Chainlit `steps."autoCollapse"` compatibility column from
  `20260505_0003_chainlit_step_autocollapse.py`.

### Secrets and production auth

For any non-local deployment:

- replace `CHAINLIT_AUTH_SECRET` with a long random secret;
- replace `CHAINLIT_AUTH_USERNAME` / `CHAINLIT_AUTH_PASSWORD` or implement a
  production auth callback/provider;
- do not commit `.env`;
- provide model provider secrets through the deployment platform secret store.

### Chainlit Socket.IO and Redis

Chainlit serves browser realtime traffic under `/ws/socket.io`. The current app
uses Redis for app/tool caching only. Chainlit 2.11.1 initializes Socket.IO
without an `AsyncRedisManager`, so Redis does **not** provide Chainlit
multi-replica websocket/polling state.

Current deployment recommendation:

- run a single app replica; or
- if deploying multiple replicas later, design sticky sessions, WebSocket-only
  transport, or upstream-supported Socket.IO Redis-manager integration in a
  separate architecture change.

### Uploads/elements

No Chainlit storage provider is configured. SQL persistence covers text
messages/steps/history, not durable binary elements. If production uploads are
required, add a supported storage provider and test element persistence before
raising upload size/type limits.

## Troubleshooting

### Probe query-kit providers without Docker, Chainlit, or an LLM

Use the app adapter probe before debugging browser/UI issues:

```bash
QUERY_KIT_PROVIDERS=semantic-scholar,semantic-scholar-web,pubmed,arxiv-web \
QUERY_KIT_TIMEOUT_SECONDS=30 \
QUERY_CLI_USER_AGENT='indic-research-agent/0.1 (+https://github.com/neeraj1909/indic-research-agent)' \
uv run python scripts/probe_querykit_providers.py --query "Hindi OCR" --limit 3 --since-year 2020
```

The probe prints normalized app `SearchResult` payloads, provider names, elapsed
time, and full available snippets/abstracts. It does not call an LLM and does
not require Docker or Chainlit.

For Phoenix trace debugging, add Phoenix env vars and a stable session id:

```bash
PHOENIX_ENABLED=true \
PHOENIX_COLLECTOR_ENDPOINT=http://10.20.30.1:16006 \
PHOENIX_PROJECT_NAME=indic-research-agent \
QUERY_KIT_PROVIDERS=semantic-scholar,semantic-scholar-web,pubmed,arxiv-web \
QUERY_KIT_TIMEOUT_SECONDS=30 \
QUERY_CLI_USER_AGENT='indic-research-agent/0.1 (+https://github.com/neeraj1909/indic-research-agent)' \
uv run python scripts/probe_querykit_providers.py \
  --query "Hindi OCR" --limit 3 --since-year 2020 --session-id querykit-debug-1
```

Then inspect the latest provider spans:

```bash
curl -fsS 'http://10.20.30.1:16006/v1/projects/indic-research-agent/spans?limit=80' \
  | jq '.data[] | select(.name|startswith("query-kit.provider")) | {name, parent_id, status_code, attributes}'
```

Expected useful signals:

- one root `probe.query_kit` / `agent.run` trace for a request, with provider
  spans as children rather than many disconnected traces;
- provider name, query, `since_year`, limit, elapsed milliseconds, result count,
  status, failure category, and HTTP status where available;
- no API keys, cookies, auth headers, or copied browser fingerprint headers in
  logs/spans.

For local visual Phoenix inspection, use `cdp` against
`http://10.20.30.1:16006/projects/UHJvamVjdDoz/traces` and keep raw captures
under `tmp/`.

### `You didn't provide an API key` or model auth errors

Check which model variables are visible in the app container without printing
secret values:

```bash
docker compose exec -T app sh -lc 'for name in LITELLM_MODEL LITELLM_API_KEY LITELLM_API_BASE OPENAI_API_KEY ANTHROPIC_API_KEY; do eval value=${$name-}; if [ -n "$value" ]; then echo "$name=SET"; else echo "$name=EMPTY"; fi; done'
```

After changing `.env`, recreate the app container:

```bash
docker compose up -d --force-recreate app
```

Run a live model smoke:

```bash
timeout 25 docker compose exec -T app uv run --no-sync python -m indic_research_agent.agent.llm --smoke --live --prompt "Reply with OK."
```

If this fails, the issue is likely model/provider configuration rather than
Chainlit.

### Chainlit login does not appear or `/auth/config` has auth disabled

Verify that the app imported `chainlit_app.py` and that auth env vars are set:

```bash
curl -fsS http://localhost:8000/auth/config | jq .
docker compose exec -T app sh -lc 'for name in CHAINLIT_AUTH_SECRET CHAINLIT_AUTH_ENABLED CHAINLIT_AUTH_USERNAME CHAINLIT_AUTH_PASSWORD; do eval value=${$name-}; if [ -n "$value" ]; then echo "$name=SET"; else echo "$name=EMPTY"; fi; done'
```

Expected for local auth:

```json
{
  "requireLogin": true,
  "passwordAuth": true
}
```

### Login succeeds but history is not resumable

Check Chainlit project settings:

```bash
curl -fsS http://localhost:8000/project/settings?language=en-US | jq '{dataPersistence,threadResumable}'
```

Expected:

```json
{
  "dataPersistence": true,
  "threadResumable": true
}
```

If false, verify:

- `CHAINLIT_DATABASE_URL` is set and reachable;
- `CHAINLIT_DATABASE_SCHEMA=chainlit`;
- migrations ran successfully;
- `@cl.data_layer` and `@cl.on_chat_resume` are still registered in
  `src/indic_research_agent/ui/chainlit_app.py`.

### Database connection or missing table errors

Run migrations explicitly:

```bash
APP_DATABASE_URL='postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent' \
uv run alembic upgrade head
```

Inspect key tables:

```bash
docker compose exec -T postgres psql -U postgres -d indic_research_agent -c \
  "select to_regclass('public.queries'), to_regclass('chainlit.users'), to_regclass('chainlit.steps');"
```

### Redis/cache failures

Check Redis health:

```bash
docker compose exec -T redis redis-cli ping
```

Run the Redis integration test:

```bash
REDIS_URL='redis://localhost:6379/0' uv run pytest tests/integration/test_redis_cache.py -q
```

### No progress or no streaming tokens in the UI

Progress steps should appear even when token-level streaming is unavailable. If
no progress appears:

- confirm `.chainlit/config.toml` has `cot = "full"`;
- check app logs for `agent.llm.start`, `agent.tool.start`, or
  `agent.stream.failed`;
- run streaming unit/e2e smoke tests:

```bash
uv run pytest tests/unit/services/test_agent_service_streaming.py tests/e2e/test_chainlit_streaming_smoke.py -q
```

If progress appears but tokens do not stream, the selected model/adapter may be
returning a completed `AIMessage` rather than incremental chunks. The UI should
still update with the final answer.

### Query-kit provider failures

`QueryKitService` retries individual fallback providers when combined provider
search raises `ProviderSearchError`. Check logs for provider-specific warnings.
Provider-specific credentials, quotas, and network requirements are not
currently documented in this repository.

### Slow answers

The default prompt asks the agent to search both local BM25 and public research
providers. Public provider calls can dominate latency. To narrow search, the LLM
must choose or be prompted toward `source="local"`; there is not currently a UI
setting for source selection.

## Security notes

- `.env` is ignored by git; do not commit secrets.
- `CHAINLIT_AUTH_SECRET` signs Chainlit auth cookies. Use a strong unique value
  outside local/e2e.
- `test` / `test1234` is a local/e2e convenience only.
- Chainlit `persist_user_env = false`; user-supplied environment variables are
  not intended to be persisted.
- `.chainlit/config.toml` currently has `allow_origins = ["*"]`; restrict this
  before exposing the app broadly.
- `unsafe_allow_html = false`; keep it disabled unless there is a reviewed need.
- Chat messages, Chainlit steps, thread metadata, app query text, tool arguments,
  tool summaries, and agent answers may be persisted in PostgreSQL. Treat the
  database as sensitive user/research data.
- Uploaded files are not durably stored by a configured object storage provider,
  but users should still avoid uploading secrets or sensitive documents until a
  reviewed ingestion/storage policy exists.
- The no-vector-dependency guard helps preserve the BM25-first retrieval policy.

## Contributing

No formal branching or pull-request policy is currently documented in the repo.
Recent commits use concise conventional-commit-style messages, but this is not
encoded as an automated rule.

Suggested workflow:

1. Create a topic branch.
2. Make focused changes with tests.
3. Run formatting/linting and the relevant test tiers.
4. Keep Chainlit UI code thin; put orchestration in controllers/services/agent
   modules.
5. Keep retrieval BM25/keyword-first unless a future PRP explicitly changes the
   architecture.
6. Do not add vector/embedding dependencies without updating policy/tests.
7. Open a PR with validation output and any migration/deployment notes.

Suggested PR checklist:

- [ ] `uv run ruff format . && uv run ruff check --fix .`
- [ ] `timeout 60 uv run pytest -m unit --no-cov`
- [ ] Integration tests run when persistence/cache code changed.
- [ ] E2E smoke run when Chainlit/auth/streaming/runtime code changed.
- [ ] Alembic migration added for DB schema changes.
- [ ] README/docs updated for configuration or operational changes.
- [ ] No secrets committed.

## License

No license file was found in this repository at the time this README was
written. Treat the project as unlicensed unless the repository owner adds a
license file or otherwise documents licensing terms.
