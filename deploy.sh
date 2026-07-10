#!/usr/bin/env bash
set -euo pipefail

SUBMODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOTFILES_ROOT="$(cd "$SUBMODULE_DIR/../.." && pwd)"
RUNTIME_DIR="${KILO_PROXY_RUNTIME_DIR:-$HOME/.local/share/kilo-claude-proxy}"
CONFIG_DEST_DIR="${KILO_PROXY_CONFIG_DIR:-$HOME/.config/kilo-claude-proxy}"
AGENT_CONFIG_SRC="${KILO_PROXY_AGENT_CONFIG:-$DOTFILES_ROOT/config/agent-runtime/agent-config.yml}"
BIN_DIR="$RUNTIME_DIR/bin"
LOCAL_BIN_LINK="${HOME}/.local/bin/open-llm-proxy"

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
mkdir -p "$BIN_DIR"
ln -snf "$RUNTIME_DIR/.venv/bin/open-llm-proxy" "$BIN_DIR/open-llm-proxy"
if [ -e "$LOCAL_BIN_LINK" ] && [ ! -L "$LOCAL_BIN_LINK" ]; then
  echo "${LOCAL_BIN_LINK} exists and is not a symlink; refusing to overwrite" >&2
  exit 1
fi
ln -snf "$BIN_DIR/open-llm-proxy" "$LOCAL_BIN_LINK"

# 6. Ask for provider plans on first setup and initialize the SQLite policy cache.
echo "Configuring provider rate-limit plans..."
"$RUNTIME_DIR/.venv/bin/python" -m open_llm_proxy.setup \
  --config "$CONFIG_DEST_DIR/agent-config.yml"

echo "=== Deployment complete ==="
echo "Reminder to restart service:"
echo "launchctl kickstart -k gui/\$(id -u)/com.user.kilo-claude-proxy"
