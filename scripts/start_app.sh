#!/usr/bin/env sh
set -eu

sh scripts/migrate.sh
exec uv run --no-sync chainlit run src/indic_research_agent/ui/chainlit_app.py \
  --headless \
  --host 0.0.0.0 \
  --port 8000
