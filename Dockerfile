FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md alembic.ini chainlit.md ./
COPY migrations ./migrations
COPY scripts ./scripts
COPY src ./src

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["sh", "scripts/start_app.sh"]
