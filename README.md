# indic-research-agent

BM25-first AI research agent using query-kit, LiteLLM, LangGraph, Chainlit,
PostgreSQL, Redis, SQLAlchemy, and Docker.

The active implementation plan is:

`/home/neeraj/prp-plans/indic-research-agent/2026-05-05-bm25-query-kit-langgraph-agent-plan.md`

## Design Constraint

Retrieval starts with BM25 and keyword-based search. Do not add embeddings,
vector databases, or embedding model dependencies unless a later plan records
evidence that BM25 is insufficient.

The unit suite includes a direct-dependency guard for common vector and
embedding packages.

## Cache Policy

Redis cache namespaces have explicit TTLs in
`src/indic_research_agent/services/cache_policy.py`. Document changes invalidate
the `tool.search` and `tool.fetch` namespaces.

## Development

```bash
uv sync
uv run python -c "import indic_research_agent; import query_cli"
uv run ruff format . && uv run ruff check --fix .
timeout 60 uv run pytest -m unit --no-cov
```

See `docs/architecture.md` for the module map and current runtime flow.

## Run

Set the LiteLLM provider environment variables for your selected model, then
start the Chainlit UI:

```bash
cp .env.example .env
uv run chainlit run src/indic_research_agent/ui/chainlit_app.py -w
```

For a no-network adapter smoke check:

```bash
uv run python -m indic_research_agent.agent.llm --smoke
```

## Docker

```bash
docker compose up --build
```

The app container runs `scripts/migrate.sh` before starting Chainlit.

For a deterministic end-to-end smoke query without live LLM credentials:

```bash
docker compose up -d postgres redis
uv run alembic upgrade head
uv run python scripts/smoke_query.py --question "How does BM25 retrieval support this agent?"
docker compose down -v
```

The smoke script seeds a PostgreSQL document, builds a BM25 index, runs
LangGraph search/fetch tool calls, repeats the calls to prove Redis cache hits,
persists tool-call and response records, and prints a JSON summary.

## Verify It Works

Check the running containers:

```bash
docker compose ps
```

Expected: `app` is up, `postgres` is healthy, and `redis` is healthy. The UI is
available at:

```text
http://localhost:8000
```

Run the full local validation loop:

```bash
uv run ruff format . && uv run ruff check --fix .
timeout 60 uv run pytest -m unit --no-cov
APP_DATABASE_URL='postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent' REDIS_URL='redis://localhost:6379/0' uv run pytest -m integration
APP_DATABASE_URL='postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent' REDIS_URL='redis://localhost:6379/0' uv run pytest -m e2e
```

Run a deterministic smoke query against PostgreSQL and Redis:

```bash
APP_DATABASE_URL='postgresql+asyncpg://postgres:postgres@localhost:5432/indic_research_agent' \
REDIS_URL='redis://localhost:6379/0' \
uv run python scripts/smoke_query.py --question "How does BM25 retrieval support this agent?"
```

Expected smoke output is JSON with a non-empty `answer`, `tool_call_count` of
`2`, `persisted_tool_calls` of at least `2`, and `cache_hits` of at least `2`.

Inspect app logs:

```bash
docker compose logs -f app
```

For a live UI check, ask Chainlit:

```text
How does this agent use BM25 and query-kit?
```

Live UI answers require valid LiteLLM credentials in `.env`. Do not commit or
share `.env`.

## Troubleshooting

If the UI returns an error like `You didn't provide an API key`, verify that the
running container has the current `.env` values:

```bash
docker compose exec -T app sh -lc 'for name in LITELLM_MODEL LITELLM_API_KEY LITELLM_API_BASE; do eval value=${$name-}; if [ -n "$value" ]; then echo "$name=SET"; else echo "$name=EMPTY"; fi; done'
```

If `LITELLM_API_KEY` or `LITELLM_API_BASE` is `EMPTY`, recreate the app
container after updating `.env`:

```bash
docker compose up -d --force-recreate app
```

This reloads environment variables without printing or exposing secret values.

If chat responses are slow, check the model backend first:

```bash
timeout 25 docker compose exec -T app uv run --no-sync python -m indic_research_agent.agent.llm --smoke --live --prompt "Reply with OK."
```

If this times out or returns an upstream error, the delay is coming from the
configured LiteLLM/API-base backend rather than Chainlit. The agent's LangGraph
search tool defaults to `source="all"`, so chat answers use both local BM25
retrieval and query-kit public research search.
