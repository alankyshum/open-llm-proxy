from __future__ import annotations

import os
from pathlib import Path

from open_llm_proxy import env_creds


ENV_KEY = "OPENROUTER_API_KEY"


def _resolve_account_key(account: str | None) -> str | None:
    """Read key for *account* from a per-account secret file, or ``None``."""
    if account is None:
        return None
    try:
        from open_llm_proxy import account_registry

        ref = account_registry.resolve_secret_ref("openrouter", account)
        if isinstance(ref, Path) and ref.exists():
            secret = ref.read_bytes()
            if secret and secret.strip():
                return secret.decode().strip()
    except Exception:
        pass
    return None


def get_persisted_api_key(account: str | None = None) -> str:
    """Read the OpenRouter key from env file or per-account secret.

    When *account* is a named account with a file-backed credential,
    reads from that file.  Otherwise reads from the shared env file
    (``@default`` / legacy path).

    Raises ``RuntimeError`` if absent.
    """
    if account is not None:
        key = _resolve_account_key(account)
        if key:
            return key
        if account != "default":
            raise RuntimeError(
                f"OpenRouter account {account!r} has no stored credential"
            )
    # Fall through to shared env file (None or "default" only)
    key = _read_env_file(ENV_KEY)
    if key and key.strip():
        return key.strip()
    raise RuntimeError(f"{ENV_KEY} is absent from env file")


def get_api_key(account: str | None = None) -> str:
    """Reads OPENROUTER_API_KEY from environment, env file, or per-account file.

    *account* selects a named per-account secret.  When ``None`` uses the
    active/default credential (env var → env file).

    Raises ``RuntimeError`` if absent.
    """
    if account is not None:
        key = _resolve_account_key(account)
        if key:
            return key
        if account != "default":
            raise RuntimeError(
                f"OpenRouter account {account!r} has no stored credential"
            )
    key = os.environ.get(ENV_KEY)
    if key and key.strip():
        return key.strip()
    # Fall through to persisted (file-only) lookup
    return get_persisted_api_key(account=None)


def save_api_key(key: str) -> None:
    """Atomically set OPENROUTER_API_KEY=... in the env file (0600)."""
    env_creds.set_env_key(ENV_KEY, key)


# ---- internal helpers ---------------------------------------------------------


def _read_env_file(name: str) -> str | None:
    """Parse the env file for *name*, returning its value or ``None``.

    Ignores the shell environment — purely file-based.
    """
    import shlex
    from pathlib import Path

    env_file = env_creds._env_file()
    if not env_file.is_file():
        return None
    content = env_file.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        parsed = shlex.split(value.strip())
        if parsed and parsed[0].strip():
            return parsed[0].strip()
    return None
