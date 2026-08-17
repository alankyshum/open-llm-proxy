"""Portable configuration path resolution."""

import os
from pathlib import Path


def resolve_config_dir() -> Path:
    """Return the shared configuration directory for proxy state and files."""
    override = os.environ.get("OPEN_LLM_PROXY_CONFIG_DIR") or os.environ.get("OLP_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return config_home / "open-llm-proxy"


def find_agent_config(explicit: str | Path | None = None) -> Path:
    """Resolve agent-config.yml using the documented priority order."""
    env_config = os.environ.get("OPEN_LLM_PROXY_CONFIG")
    env_path = Path(env_config) if env_config else None
    if explicit is not None:
        # Explicit paths are returned unchecked so the caller reports the open error.
        return Path(explicit)
    if env_path is not None:
        if not env_path.is_file():
            raise FileNotFoundError(
                f"OPEN_LLM_PROXY_CONFIG points to a nonexistent file: {env_path}"
            )
        return env_path
    candidates = [
        resolve_config_dir() / "agent-config.yml",
        Path.cwd() / "agent-config.yml",
    ]
    for path in candidates:
        if path is not None and path.is_file():
            return path
    raise FileNotFoundError(
        "agent-config.yml not found. Set OPEN_LLM_PROXY_CONFIG or provide --config; "
        f"searched {', '.join(str(path) for path in candidates if path is not None)}"
    )
