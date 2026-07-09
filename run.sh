#!/usr/bin/env bash
# Convenience launcher for open-llm-proxy.
set -euo pipefail
export PYTHONUNBUFFERED=1
export BYPASS_KEYCHAIN=1

ENV_FILE="${KILO_PROXY_ENV_FILE:-$HOME/.config/kilo-claude-proxy/env}"
if [ -f "$ENV_FILE" ]; then
  while read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    _k="${line%%=*}"
    _v="${line#*=}"
    if ! printenv "$_k" >/dev/null 2>&1; then export "$_k=$_v"; fi
  done < "$ENV_FILE"
fi

echo "================================================================="
echo "open-llm-proxy — LiteLLM Router-based local proxy"
echo "Listening on http://localhost:${KILO_PROXY_PORT:-8765}"
echo "================================================================="

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

exec .venv/bin/python -m open_llm_proxy.server_launcher --port "${KILO_PROXY_PORT:-8765}" "$@"
