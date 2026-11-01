# indic-research-agent

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Chainlit](https://img.shields.io/badge/UI-Chainlit-111827)](https://chainlit.io/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-0F766E)](https://langchain-ai.github.io/langgraph/)

`indic-research-agent` is an India- and Indic-language-focused research assistant. It searches public research providers, calls OpenAI directly to synthesize evidence, and presents the answer through a Chainlit web UI.

The project is designed as a transparent research workflow: source search, tool calls, progress, citations, and uncertainty should remain visible instead of being hidden behind an unsupported retrieval claim.

## Table of contents

- [Problem statement](#problem-statement)
- [What the project does](#what-the-project-does)
- [Research workflow](#research-workflow)
- [Architecture](#architecture)
- [Running locally](#running-locally)
- [Configuration](#configuration)
- [Usage](#usage)
- [Offline Decisions replay](#offline-decisions-replay)
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
- synthesize source-backed answers with the direct OpenAI generation model;
- show live progress and streamed output in Chainlit; and
- persist chat history, audit metadata, and cache entries for local development.

The current application is a public-provider research assistant. It does not claim to search a local document corpus when no ingestion or local retrieval pipeline is configured.

## What the project does

| Capability | Current behavior |
| --- | --- |
| Indic/India research | The system prompt specializes the agent for Indic-language and India-focused research questions. |
| Public research search | `query-kit` providers are called through `QueryKitService` and the typed `search` tool. |
| Answer synthesis | The direct OpenAI chat adapter is orchestrated through LangGraph; generation defaults to `gpt-4.1-mini`. |
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
  -> direct OpenAI generation (gpt-4.1-mini)
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

- `__start__ -> llm`: begin with the direct OpenAI generation model.
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
- An `OPENAI_API_KEY` with access to the configured generation model for live answers

### 1. Clone and install

```bash
git clone git@github.com:neeraj1909/indic-research-agent.git
cd indic-research-agent

uv sync
cp .env.example .env
```

Edit `.env` before starting the application: set `OPENAI_API_KEY` privately and replace the development authentication secret for any non-local deployment.

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
| `OPENAI_API_KEY` | Credential for direct OpenAI generation and Decisions; keep it private. |
| `OPENAI_MODEL` | Generation model; defaults to `gpt-4.1-mini`. |
| `OPENAI_TEMPERATURE` / `OPENAI_MAX_TOKENS` | Generation sampling and output limit. |
| `OPENAI_TIMEOUT_SECONDS` / `OPENAI_STREAMING` | Generation transport timeout and streaming flag. |
| `DECISIONS_MODE` / `DECISIONS_MODEL` | Optional routing experiment; defaults to `off` and `gpt-6-luna`. |
| `DECISIONS_TIMEOUT_SECONDS` / `DECISIONS_MAX_CONCURRENCY` | Routing timeout and concurrency bounds. |
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

## Offline Decisions replay

### What OpenAI Decisions is

[OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions) is a
typed evaluation API for bounded judgments over text, images, or JSON. It is a
candidate for one small routing decision; it is not a replacement for this
project's generative model, public-research tools, LangGraph orchestration, or
final answer synthesis. The experiment asks whether a separate decision can
help classify a request as `research`, `clarify`, `supplied_text`, or
`unsupported_local_documents` while the existing workflow remains authoritative.

### Status: direct OpenAI generation and Decisions

| Surface | Current behavior |
| --- | --- |
| Research workflow | Chainlit -> AgentService/LangGraph -> direct OpenAI generation -> typed `query-kit` public search -> cited answer. PostgreSQL and Redis retain their existing roles. |
| Generation | Defaults to `gpt-4.1-mini` at `https://api.openai.com/v1/chat/completions`, including the active `supplied_text` transformation path. |
| Decisions | Uses `gpt-6-luna` at `https://api.openai.com/v1/decisions`; mode defaults to `off`. Application `shadow`/`active` modes and opt-in evaluator CLIs share the direct transport. |
| Transport isolation | Both transports receive an explicit OpenAI key, ignore inherited proxy/base-URL settings, disable redirects, and make no automatic retries. LiteLLM is not called by these application or comparison paths; the configured LiteLLM generation route initiated an unexpected device authorization flow during validation. |
| Evidence limit | Endpoint access, controlled-response tests, and offline replay do not establish a successful 20-case paired run or improved answer quality. Consult each run's artifacts. |

The existing `LITELLM_*` settings remain accepted as legacy configuration, but
these direct generation/Decisions paths do not use them. `OPENAI_BASE_URL`
does not redirect either transport. Generation and Decisions are separate
operations: the routing endpoint does not generate answers or search arguments.
Retrieval remains keyword/BM25-first through `query-kit` (`query_cli`).

The offline replay is separate from Chainlit and uses no credentials, network,
live model, or paid fallback. Live generation and Decisions calls may incur
OpenAI charges; an existing subscription alone does not establish API access.

The 60-case request fixture at
[`tests/fixtures/decisions/requests.jsonl`](tests/fixtures/decisions/requests.jsonl)
contains machine-drafted labels only. All cases require actual human review
before semantic scoring or enforcement. The separate
[`controlled-responses.jsonl`](tests/fixtures/decisions/controlled-responses.jsonl)
file contains local response fixtures for wiring and fallback checks; these are
not model outputs and do not establish model quality.

Run with the exact evaluator arguments:

```bash
# Off: zero decision-provider calls; every case records the legacy route.
PHOENIX_ENABLED=false uv run --no-sync python scripts/evaluate_decisions.py \
  --offline \
  --cases tests/fixtures/decisions/requests.jsonl \
  --split all \
  --mode off \
  --output tmp/decisions-implementation/replay-off.json

# Shadow: evaluate separate local controlled responses, outside AgentService.
PHOENIX_ENABLED=false uv run --no-sync python scripts/evaluate_decisions.py \
  --offline \
  --cases tests/fixtures/decisions/requests.jsonl \
  --responses tests/fixtures/decisions/controlled-responses.jsonl \
  --split all \
  --mode shadow \
  --output tmp/decisions-implementation/replay.json
```

The first command is the smallest offline smoke path: it needs only the checked-
in fixtures and writes a local replay ledger. The second command exercises the
same contract with checked-in controlled responses; it is still not a live
model evaluation. Neither command changes the user-visible Chainlit route.

### Application behavior by Decisions mode

`DECISIONS_MODE` defaults to `off` and accepts only `off`, `shadow`, or
`active`.

| Mode | Actual application behavior |
| --- | --- |
| `off` | Makes no Decisions-provider call. The existing LangGraph research graph remains authoritative, including direct OpenAI generation and query-kit search. |
| `shadow` | Makes one bounded decision attempt when a provider is configured, records sanitized status/route/fallback information in tracing, then deliberately keeps the existing research graph authoritative. A timeout, provider error, or invalid answer becomes the conservative `legacy` fallback; it does not change the user-visible route. |
| `active` | Uses one bounded decision attempt. `research` and `legacy` fallback continue through the research graph; `clarify` returns a deterministic clarification without graph/search; `unsupported_local_documents` returns the deterministic local-document limitation without graph/search; `supplied_text` uses the separately configured generation model's direct OpenAI transformation call without the research search tool. The Decisions-only model call is not used for generation, and generation support must be verified independently. A valid active decision is surfaced as sanitized progress metadata. |

The default decision timeout is 5 seconds (bounded to 10 seconds) and default
decision concurrency is one. The active route is therefore an experiment, not a
claim that Decisions improves answer quality. Local cancellation or timeout
does not prove that OpenAI stopped processing a request; it may still incur
provider cost.

### Paired FRAMES comparison

The repository comparison is a paired ablation of the routing decision. Both
arms receive the same question, pinned dataset row, requested model, application
prompt/configuration, retrieval/provider settings, tool budget, and grading
procedure. Each pair is counterbalanced by a deterministic hash so the arm order
alternates reproducibly. The arms are:

- `no_decisions`: `DECISIONS_MODE=off`, the legacy research path.
- `with_decisions`: `DECISIONS_MODE=active`, with the Decisions provider injected
  into `AgentService`.

The shared dataset is the first 20 rows of the official Google FRAMES test split,
stored at [`data/evaluations/frames/frames-test-20.jsonl`](data/evaluations/frames/frames-test-20.jsonl).
It is pinned to revision `58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`, preserves
the original English questions, reference answers, and Wikipedia evidence URLs,
and reports the official Apache-2.0 license. See the [FRAMES dataset card](https://huggingface.co/datasets/google/frames-benchmark),
[pinned source TSV](https://huggingface.co/datasets/google/frames-benchmark/resolve/58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef/test.tsv),
and the [local slice provenance](data/evaluations/frames/README.md).

This is an English-only, Wikipedia-grounded, nonrepresentative 20-row slice. It
is not evidence of broad Indic-language quality, and the current providers may
not retrieve the referenced Wikipedia evidence. Live retrieval is therefore
unfrozen and can vary; model generation is also stochastic. Results can conflate
routing, provider coverage, retrieval, and reasoning. Exact normalized answer
match and normalized reference-contained-in-answer are strict automatic proxies,
not semantic truth. Non-match, uncertainty, failed runs, and citation quality
require human review; arm equivalence is not correctness. Report per-arm route,
status/fallback, provider failures, tool calls, citations, latency, usage when
reported, and billing evidence separately.

Dry-run the pinned input without making model, Decisions, or retrieval calls:

```bash
uv run python scripts/evaluate_agent_comparison.py \
  --frames data/evaluations/frames \
  --limit 20 \
  --repetitions 1 \
  --output-dir tmp/decisions-live/comparison-dry-run
```

Run a bounded live pair only after authorizing the provider calls and setting
`OPENAI_API_KEY` privately in the environment or untracked `.env`. Both arms use
`gpt-4.1-mini` for generation; only the routing-decision mode differs. The active
arm adds one `gpt-6-luna` Decisions attempt per valid case. Public retrieval uses
`query-kit`; no LiteLLM or local model is used.

```bash
PHOENIX_ENABLED=false uv run --env-file .env python scripts/evaluate_agent_comparison.py \
  --live \
  --frames data/evaluations/frames \
  --limit 1 --repetitions 1 \
  --model gpt-6-luna \
  --generation-model gpt-4.1-mini \
  --api-key-env OPENAI_API_KEY \
  --generation-api-key-env OPENAI_API_KEY \
  --output-dir tmp/decisions-live/direct-openai-one-pair
```

After checking that pair's artifacts, change `--limit 1` to `--limit 20` and use
a fresh output directory for the complete checked-in slice. This command is an
execution recipe, not a claim that all 20 cases have succeeded. The CLI permits
limits of 1–50 and 1–5 repetitions, bounded by the available input rows. It stops
after a failed pair and checkpoints records, so count completed pairs explicitly.
Generation can make several calls per research case as the graph uses tools;
`--limit` is not a total API-request or spending cap.

The live command writes `manifest.json`, `records.jsonl`, `pairs.jsonl`, and
`summary.json`. Check requested generation/decision identities, per-arm status,
route, provider failures, citations, latency, usage when available, and the
actual record count. Missing generation usage or billing evidence remains
unknown. Exact-match and containment scores are automatic proxies; human
review is required before any quality conclusion.

### Direct OpenAI result (2026-11-01)

The bounded run completed **20 pairs / 40 application records** using
`gpt-4.1-mini` generation and `gpt-6-luna` Decisions. OFF made zero Decisions
requests; ACTIVE made 20, all selecting `research`.

| Observation | OFF | ACTIVE |
| --- | --- | --- |
| Completed records | 20/20 | 20/20 |
| Exact reference match | 0/20 | 0/20 |
| Permissive reference containment | 3/20 | 3/20 |
| Records with citation markers | 12/20 | 10/20 |

Eighteen pairs require human review. Citation markers do not establish citation
correctness, and permissive containment is not semantic correctness. Academic
provider retrieval was unfrozen, and generation usage/cost was untracked;
these results support no causal or answer-quality improvement conclusion.

The local [run summary](tmp/decisions-live/direct-openai-frames20-20261101/summary.json)
and adjacent `records.jsonl` are ignored, ephemeral artifacts: they are available
in this workspace, not guaranteed to exist in a fresh clone.

### Decisions comparison credentials and settings

```text
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TEMPERATURE=0.2
OPENAI_TIMEOUT_SECONDS=30
OPENAI_MAX_TOKENS=1200
OPENAI_STREAMING=true
DECISIONS_MODE=off
DECISIONS_MODEL=gpt-6-luna
DECISIONS_TIMEOUT_SECONDS=5.0
DECISIONS_MAX_CONCURRENCY=1
QUERY_KIT_PROVIDERS=semantic-scholar,semantic-scholar-web,pubmed,arxiv-web
```

Set `OPENAI_API_KEY` privately; never paste it into commands, reports, or issue
trackers. The application reads this key for both direct transports. The paired
CLI can select credential variable names with `--api-key-env` and
`--generation-api-key-env`; its explicit `--generation-model` selects generation
independently of `--model`, which selects Decisions. Existing process variables
can override `.env` values. `DECISIONS_API_KEY` and legacy proxy credentials are
not the current application's credential contract.

Docker Compose forwards `OPENAI_API_KEY`, the `OPENAI_*` generation settings,
and the `DECISIONS_*` routing settings. Its defaults match the local application:
`gpt-4.1-mini` generation and Decisions mode `off`. Rebuild/recreate the app
container after changing its configuration; a local code edit does not prove
the running container has those changes.

### Historical failed comparison evidence

Earlier proxy/LiteLLM attempts recorded one pair (two failed generation arms),
including a run whose active Decisions request succeeded with 185 total tokens.
They produced no completed answer-quality or equivalence pair. Separate
historical probes reported native HTTP `401`, proxy connection refusal, and a
private-endpoint `4xx`; those observations are distinct and do not identify the
cause of current direct-route failures. Proxy recovery instructions and
LiteLLM upgrade commands from those attempts do not describe the current
transport. Keep historical artifacts separate from new direct OpenAI results;
neither endpoint success nor offline replay is a paired quality result.

### One-case live CLI (explicit opt-in)

The documented Decisions model is exactly `gpt-6-luna`. To run exactly one
valid fixture case, provide the `OPENAI_API_KEY` environment prerequisite at
runtime. Do not commit or print its value:

```bash
export OPENAI_API_KEY
uv run python scripts/evaluate_decisions_live.py \
  --live \
  --case-id dev-en-18 \
  --cases tests/fixtures/decisions/requests.jsonl \
  --model gpt-6-luna \
  --output tmp/decisions-live/dev-en-18.json
```

The CLI requires `--live`, selects exactly one valid case, defaults to
`gpt-6-luna`, and makes one direct Decisions request without automatic retries.
It may incur charges. This isolated transport probe does not run generation,
public retrieval, or a paired comparison, and proves no answer-quality result.

For the offline `evaluate_decisions.py` replay, use `--split dev`,
`--split development`, or `--split heldout` to select a subset. These flags do
not apply to the one-case live CLI. The replay ledger reports controlled-provider
calls, not actual model
calls; resolved model identity remains unknown. See
[`docs/evaluations/decisions-results.md`](docs/evaluations/decisions-results.md)
for interpretation limits.

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
- [OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions)
- [OpenAI Agents SDK README](https://github.com/openai/openai-agents-python)
- [LangGraph README](https://github.com/langchain-ai/langgraph)
- [OpenAI Python SDK README](https://github.com/openai/openai-python)
- [`query-kit`](https://github.com/neeraj1909/query-kit)

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
