#!/usr/bin/env bash
set -euo pipefail

SUBMODULE_DIR="/Users/alanshum/Documents/dotfiles/.external/open-llm-proxy"
RUNTIME_DIR="/Users/alanshum/.local/share/kilo-claude-proxy"
CONFIG_DEST_DIR="/Users/alanshum/.config/kilo-claude-proxy"
AGENT_CONFIG_SRC="/Users/alanshum/Documents/dotfiles/config/agent-runtime/agent-config.yml"

echo "=== Deploying open-llm-proxy submodule -> runtime ==="

# 1. Sync the package directory (open_llm_proxy/)
echo "Syncing open_llm_proxy/ package..."
mkdir -p "$RUNTIME_DIR/open_llm_proxy"
rsync -av --delete "$SUBMODULE_DIR/open_llm_proxy/" "$RUNTIME_DIR/open_llm_proxy/"

# 2. Sync run.sh
if [ -f "$RUNTIME_DIR/run.sh" ]; then
  if ! cmp -s "$SUBMODULE_DIR/run.sh" "$RUNTIME_DIR/run.sh"; then
    echo "run.sh changed. Backing up existing to run.sh.kilo-bak..."
    cp "$RUNTIME_DIR/run.sh" "$RUNTIME_DIR/run.sh.kilo-bak"
    cp "$SUBMODULE_DIR/run.sh" "$RUNTIME_DIR/run.sh"
    chmod +x "$RUNTIME_DIR/run.sh"
  fi
else
  echo "Copying run.sh to runtime..."
  cp "$SUBMODULE_DIR/run.sh" "$RUNTIME_DIR/run.sh"
  chmod +x "$RUNTIME_DIR/run.sh"
fi

# 3. Sync config yml
echo "Syncing agent-config.yml..."
mkdir -p "$CONFIG_DEST_DIR"
cp "$AGENT_CONFIG_SRC" "$CONFIG_DEST_DIR/agent-config.yml"

# 4. Copy pyproject.toml to runtime for dependency checks
cp "$SUBMODULE_DIR/pyproject.toml" "$RUNTIME_DIR/pyproject.toml"

# 5. Check and install dependencies in runtime's venv
echo "Checking dependencies in runtime's .venv..."
if [ -d "$RUNTIME_DIR/.venv" ]; then
  "$RUNTIME_DIR/.venv/bin/pip" install -q --disable-pip-version-check "$RUNTIME_DIR"
else
  echo "Creating runtime .venv..."
  python3 -m venv "$RUNTIME_DIR/.venv"
  "$RUNTIME_DIR/.venv/bin/pip" install -q --disable-pip-version-check "$RUNTIME_DIR"
fi

echo "=== Deployment complete ==="
echo "Reminder to restart service:"
echo "launchctl kickstart -k gui/\$(id -u)/com.user.kilo-claude-proxy"
