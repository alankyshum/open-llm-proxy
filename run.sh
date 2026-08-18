#!/usr/bin/env bash
# Convenience launcher for open-llm-proxy.
set -euo pipefail
export PYTHONUNBUFFERED=1
export BYPASS_KEYCHAIN=1

ENV_FILE="${OPEN_LLM_PROXY_ENV_FILE:-$HOME/.config/open-llm-proxy/env}"
if [ -f "$ENV_FILE" ]; then
  while read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    _k="${line%%=*}"
    _v="${line#*=}"
    if ! printenv "$_k" >/dev/null 2>&1; then export "$_k=$_v"; fi
  done < "$ENV_FILE"
fi

# The proxy is intentionally loopback-only; Tailscale Serve is the external path.
OPEN_LLM_PROXY_HOST="127.0.0.1"

echo "================================================================="
echo "open-llm-proxy — LiteLLM Router-based local proxy"
echo "Listening on http://${OPEN_LLM_PROXY_HOST}:${OPEN_LLM_PROXY_PORT:-8765}"
echo "================================================================="

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Ensure the Prisma client is generated when a database is configured (Admin UI).
# Idempotent + self-healing: regenerates after a redeploy recreates the venv.
if [ -n "${DATABASE_URL:-}" ]; then
  SCHEMA="$(.venv/bin/python -c 'import os,litellm; print(os.path.join(os.path.dirname(litellm.__file__),"proxy","schema.prisma"))' 2>/dev/null || true)"
  if [ -n "$SCHEMA" ] && [ -f "$SCHEMA" ]; then
    if ! .venv/bin/python -c 'import prisma.client' >/dev/null 2>&1; then
      echo "Generating Prisma client for the Admin UI..."
      PATH="$HERE/.venv/bin:$PATH" .venv/bin/python -m prisma generate --schema "$SCHEMA" || true
    fi
    echo "Syncing database schema (prisma db push)..."
    PATH="$HERE/.venv/bin:$PATH" .venv/bin/python -m prisma db push --schema "$SCHEMA" --skip-generate --accept-data-loss || true
  fi
fi

exec .venv/bin/python -m open_llm_proxy.server_launcher --host "${OPEN_LLM_PROXY_HOST}" --port "${OPEN_LLM_PROXY_PORT:-8765}" "$@"
