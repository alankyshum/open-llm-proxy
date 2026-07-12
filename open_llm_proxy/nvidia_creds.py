from __future__ import annotations

from pathlib import Path

from open_llm_proxy import env_creds


ENV_KEY = "NVIDIA_API_KEY"


def _resolve_account_key(account: str | None) -> str | None:
    """Read key for *account* from a per-account secret file, or ``None``."""
    if account is None:
        return None
    try:
        from open_llm_proxy import account_registry

        ref = account_registry.resolve_secret_ref("nvidia", account)
        if isinstance(ref, Path) and ref.exists():
            secret = ref.read_bytes()
            if secret and secret.strip():
                return secret.decode().strip()
    except Exception:
        pass
    return None


def save_api_key(key: str) -> None:
    """Persist *key* to the shared env file as NVIDIA_API_KEY=..."""
    env_creds.set_env_key(ENV_KEY, key)


def get_api_key(account: str | None = None) -> str:
    """Resolve NVIDIA_API_KEY from env, env file, or per-account file.

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
                f"NVIDIA account {account!r} has no stored credential"
            )
    key = env_creds.get_env_key(ENV_KEY)
    if key and key.strip():
        return key.strip()
    raise RuntimeError(f"{ENV_KEY} is absent")
