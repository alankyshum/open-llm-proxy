from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("open_llm_proxy.opencode_creds")


class OpenCodeAuthError(RuntimeError):
    """No usable OpenCode credential available."""


def _get_opencode_auth_path() -> Path:
    path_str = os.environ.get("OPENCODE_AUTH_PATH")
    if path_str:
        return Path(path_str)
    return Path.home() / ".local" / "share" / "opencode" / "auth.json"


def get_opencode_api_key() -> str:
    """
    Retrieve the OpenCode API key securely.
    
    Highest priority is the environment variable OPENCODE_API_KEY.
    Fallback is the ~/.local/share/opencode/auth.json file.
    """
    env_key = os.environ.get("OPENCODE_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    path = _get_opencode_auth_path()
    if not path.is_file():
        raise OpenCodeAuthError(
            f"No OpenCode API key found. Please set OPENCODE_API_KEY environment variable "
            f"or configure it in '{path}'."
        )

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        raise OpenCodeAuthError(
            f"Failed to read OpenCode auth file at '{path}': Unreadable file. "
            f"{e.__class__.__name__}: {e.strerror or str(e)}"
        )

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise OpenCodeAuthError(
            f"Failed to parse OpenCode auth file at '{path}': Malformed JSON. "
            f"Please ensure it is valid JSON. {e.__class__.__name__}: {str(e)}"
        )

    if not isinstance(data, dict):
        raise OpenCodeAuthError(
            f"Failed to parse OpenCode auth file at '{path}': Expected a JSON object."
        )

    opencode_sec = data.get("opencode")
    if not isinstance(opencode_sec, dict):
        raise OpenCodeAuthError(
            f"Failed to parse OpenCode auth file at '{path}': Missing 'opencode' section."
        )

    key = opencode_sec.get("key")
    if not isinstance(key, str) or not key.strip():
        raise OpenCodeAuthError(
            f"Failed to parse OpenCode auth file at '{path}': Missing or invalid 'key' in 'opencode' section."
        )

    return key.strip()
