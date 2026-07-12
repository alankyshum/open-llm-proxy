from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ACCOUNT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
MAX_ACCOUNT_NAME_LEN = 32

_EXT_MAP = {
    "claude-oauth": "credentials.json",
    "copilot-oauth": "json",
    "api-key": "key",
    "env-line": "key",
}


class AccountRegistryError(RuntimeError):
    """Raised on invalid registry operations."""


def _storage_ext(storage: str) -> str:
    return _EXT_MAP.get(storage, "secret")


def CONFIG_DIR() -> Path:
    """Config directory, respecting OLP_CONFIG_DIR env override."""
    override = os.environ.get("OLP_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".config" / "open-llm-proxy"


def _registry_path() -> Path:
    return CONFIG_DIR() / "accounts.json"


def _accounts_dir() -> Path:
    return CONFIG_DIR() / "accounts"


def normalize_account_name(name: str) -> str:
    """Lowercase and validate account name."""
    normalized = name.lower()
    if not normalized:
        raise AccountRegistryError("Account name must not be empty")
    if len(normalized) > MAX_ACCOUNT_NAME_LEN:
        raise AccountRegistryError(
            f"Account name must be at most {MAX_ACCOUNT_NAME_LEN} characters"
        )
    if not ACCOUNT_NAME_RE.match(normalized):
        raise AccountRegistryError(
            f"Invalid account name {name!r}: must match ^[a-z0-9][a-z0-9_-]*$"
        )
    return normalized


@dataclass
class AccountInfo:
    provider: str
    name: str
    created_at: str
    storage: str
    ref: str
    is_active: bool


# ---- Internal I/O ----------------------------------------------------------------

def _read_registry() -> dict:
    """Read accounts.json, returning empty skeleton if file missing."""
    path = _registry_path()
    if not path.is_file():
        return {"version": 1, "providers": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise AccountRegistryError(f"Failed to read registry: {e}") from e
    if not isinstance(data, dict) or data.get("version") != 1:
        raise AccountRegistryError("Invalid or unsupported registry format")
    return data


def _write_registry(data: dict) -> None:
    """Atomically write registry JSON with 0600 perms."""
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(str(path.parent), 0o700)
    except Exception as e:
        raise AccountRegistryError(
            f"Failed to set permissions on config directory: {e}"
        ) from e
    fd, tmp_str = tempfile.mkstemp(dir=str(path.parent), prefix="accounts_tmp_")
    tmp_path = Path(tmp_str)
    try:
        os.chmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tmp_path.replace(path)
        try:
            os.chmod(str(path), 0o600)
        except Exception as e:
            raise AccountRegistryError(
                f"Failed to set permissions on registry file: {e}"
            ) from e
    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise e


def _write_secret_file(path: Path, content: bytes) -> None:
    """Write secret bytes to path atomically with 0600 perms."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(str(path.parent), 0o700)
    except Exception as e:
        raise AccountRegistryError(
            f"Failed to set permissions on secrets directory: {e}"
        ) from e
    fd, tmp_str = tempfile.mkstemp(dir=str(path.parent), prefix="secret_tmp_")
    tmp_path = Path(tmp_str)
    try:
        os.chmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        tmp_path.replace(path)
        try:
            os.chmod(str(path), 0o600)
        except Exception as e:
            raise AccountRegistryError(
                f"Failed to set permissions on secret file: {e}"
            ) from e
    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise e


def _secret_rel_path(provider: str, name: str, storage: str) -> str:
    ext = _storage_ext(storage)
    return f"accounts/{provider}/{name}.{ext}"


# ---- Public API ------------------------------------------------------------------

def load() -> dict:
    """Read and return the full registry dict."""
    return _read_registry()


def list_providers() -> list[str]:
    data = _read_registry()
    return sorted(data["providers"].keys())


def list_accounts(provider: str) -> list[AccountInfo]:
    data = _read_registry()
    prov_data = data["providers"].get(provider)
    if prov_data is None:
        return []
    active = prov_data.get("active")
    return [
        AccountInfo(
            provider=provider,
            name=name,
            created_at=entry["created_at"],
            storage=entry["storage"],
            ref=entry["ref"],
            is_active=(name == active),
        )
        for name, entry in prov_data["accounts"].items()
    ]


def active_account(provider: str) -> str | None:
    data = _read_registry()
    prov_data = data["providers"].get(provider)
    if prov_data is None:
        return None
    return prov_data.get("active")


def add_account(
    provider: str,
    name: str | None = None,
    *,
    storage: str,
    secret_bytes: bytes | None = None,
    ref: str | None = None,
) -> AccountInfo:
    """Add an account for *provider*.

    If *name* is ``None`` and the provider has no accounts yet, the account
    is auto-named ``"default"``.  If the provider already has accounts,
    *name* is required.

    Exactly one of *secret_bytes* (written to a per-account file) or
    *ref* (stored verbatim) must be given.
    """
    data = _read_registry()
    prov_data = data["providers"].setdefault(provider, {"active": None, "accounts": {}})
    accounts = prov_data["accounts"]

    if name is None:
        if accounts:
            raise AccountRegistryError(
                "Account name required when provider already has accounts"
            )
        name = "default"
    else:
        name = normalize_account_name(name)

    if name in accounts:
        raise AccountRegistryError(
            f"Account {name!r} already exists for provider {provider!r}"
        )

    created_at = datetime.now(timezone.utc).isoformat()

    if secret_bytes is not None:
        rel = _secret_rel_path(provider, name, storage)
        _write_secret_file(CONFIG_DIR() / rel, secret_bytes)
        ref = rel
    elif ref is None:
        raise AccountRegistryError("Either secret_bytes or ref must be provided")

    accounts[name] = {"created_at": created_at, "storage": storage, "ref": ref}

    is_first = len(accounts) == 1
    if is_first:
        prov_data["active"] = name

    _write_registry(data)

    return AccountInfo(
        provider=provider,
        name=name,
        created_at=created_at,
        storage=storage,
        ref=ref,
        is_active=(name == prov_data["active"]),
    )


def rename_account(provider: str, old: str, new: str) -> None:
    """Rename an account.

    Guard: provider must have at least 2 accounts before renaming is allowed.
    Moves the secret file and fixes the *active* pointer if it referenced the
    old name.
    """
    data = _read_registry()
    prov_data = data["providers"].get(provider)
    if prov_data is None:
        raise AccountRegistryError(f"Unknown provider {provider!r}")

    accounts = prov_data["accounts"]
    old = normalize_account_name(old)
    new = normalize_account_name(new)

    if old not in accounts:
        raise AccountRegistryError(f"Account {old!r} not found for {provider!r}")
    if new in accounts:
        raise AccountRegistryError(f"Account {new!r} already exists for {provider!r}")
    if len(accounts) < 2:
        raise AccountRegistryError(
            "Cannot rename account: provider must have at least 2 accounts "
            "before renaming"
        )

    entry = accounts.pop(old)
    accounts[new] = entry

    old_ref = entry["ref"]
    if old_ref.startswith("accounts/"):
        old_path = CONFIG_DIR() / old_ref
        ext = _storage_ext(entry["storage"])
        new_rel = f"accounts/{provider}/{new}.{ext}"
        new_path = CONFIG_DIR() / new_rel
        if old_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.rename(new_path)
        entry["ref"] = new_rel

    if prov_data.get("active") == old:
        prov_data["active"] = new

    _write_registry(data)


def set_active(provider: str, name: str) -> None:
    """Set the active account for *provider*."""
    data = _read_registry()
    prov_data = data["providers"].get(provider)
    if prov_data is None:
        raise AccountRegistryError(f"Unknown provider {provider!r}")
    name = normalize_account_name(name)
    if name not in prov_data["accounts"]:
        raise AccountRegistryError(f"Account {name!r} not found for {provider!r}")
    prov_data["active"] = name
    _write_registry(data)


def remove_account(provider: str, name: str, *, force: bool = False) -> None:
    """Remove an account.

    Deleting the last remaining account requires *force=True*.
    If the removed account was active, the active pointer is moved to any
    remaining account.  If no accounts remain the provider entry is dropped.
    """
    data = _read_registry()
    prov_data = data["providers"].get(provider)
    if prov_data is None:
        raise AccountRegistryError(f"Unknown provider {provider!r}")

    accounts = prov_data["accounts"]
    name = normalize_account_name(name)
    if name not in accounts:
        raise AccountRegistryError(f"Account {name!r} not found for {provider!r}")

    if len(accounts) == 1 and not force:
        raise AccountRegistryError(
            f"Cannot remove last account for {provider!r}. "
            "Use force=True to remove."
        )

    entry = accounts.pop(name)

    ref = entry["ref"]
    if ref.startswith("accounts/"):
        fpath = CONFIG_DIR() / ref
        if fpath.exists():
            fpath.unlink()

    if prov_data.get("active") == name:
        if accounts:
            prov_data["active"] = next(iter(accounts))
        else:
            del data["providers"][provider]
            _write_registry(data)
            return

    _write_registry(data)


def resolve_secret_ref(provider: str, name: str) -> Path | str:
    """Return the absolute path for a file-based ref, or the raw ref string."""
    data = _read_registry()
    prov_data = data["providers"].get(provider)
    if prov_data is None:
        raise AccountRegistryError(f"Unknown provider {provider!r}")
    name = normalize_account_name(name)
    accounts = prov_data["accounts"]
    if name not in accounts:
        raise AccountRegistryError(f"Account {name!r} not found for {provider!r}")
    ref = accounts[name]["ref"]
    if ref.startswith("accounts/"):
        return CONFIG_DIR() / ref
    return ref


def read_secret(provider: str, name: str) -> bytes | None:
    """Read per-account secret file bytes, or None if ref is not a file."""
    data = _read_registry()
    prov_data = data["providers"].get(provider)
    if prov_data is None:
        raise AccountRegistryError(f"Unknown provider {provider!r}")
    name = normalize_account_name(name)
    accounts = prov_data["accounts"]
    if name not in accounts:
        raise AccountRegistryError(f"Account {name!r} not found for {provider!r}")
    ref = accounts[name]["ref"]
    if not ref.startswith("accounts/"):
        return None
    path = CONFIG_DIR() / ref
    if not path.exists():
        return None
    return path.read_bytes()
