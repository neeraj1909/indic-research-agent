#!/usr/bin/env sh
set -eu

COOKIE_WATCH_PID=""
CHAINLIT_PID=""

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
}

cleanup_children() {
  if [ -n "${CHAINLIT_PID}" ] && kill -0 "${CHAINLIT_PID}" 2>/dev/null; then
    kill "${CHAINLIT_PID}" 2>/dev/null || true
    wait "${CHAINLIT_PID}" 2>/dev/null || true
  fi
  if [ -n "${COOKIE_WATCH_PID}" ] && kill -0 "${COOKIE_WATCH_PID}" 2>/dev/null; then
    kill "${COOKIE_WATCH_PID}" 2>/dev/null || true
    wait "${COOKIE_WATCH_PID}" 2>/dev/null || true
  fi
}

on_exit() {
  status=$?
  trap - INT TERM EXIT
  cleanup_children
  exit "${status}"
}

trap on_exit INT TERM EXIT

sh scripts/migrate.sh

if is_truthy "${BROWSER_COOKIE_JAR_ENABLED:-true}"; then
  uv run --no-sync python scripts/sync_browser_cookies.py --watch --json &
  COOKIE_WATCH_PID=$!
fi

uv run --no-sync chainlit run src/indic_research_agent/ui/chainlit_app.py \
  --headless \
  --host 0.0.0.0 \
  --port 8000 &
CHAINLIT_PID=$!

set +e
wait "${CHAINLIT_PID}"
APP_STATUS=$?
set -e
CHAINLIT_PID=""

trap - EXIT
cleanup_children
exit "${APP_STATUS}"
