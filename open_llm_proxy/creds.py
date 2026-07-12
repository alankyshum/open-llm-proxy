from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

log = logging.getLogger("open_llm_proxy.creds")

_DEFAULT = "__default__"
_cached_key_cache: dict[str, str | None] = {}
_cached_time_cache: dict[str, float] = {}
_TTL: float = 30.0
_refresh_lock = threading.Lock()
_in_memory_oauth_cache: dict[str, dict | None] = {}

_KEYCHAIN_SERVICE_PRIMARY = "Claude Code"
_KEYCHAIN_SERVICE_CREDENTIALS = "Claude Code-credentials"


def _from_env() -> str | None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip()
    return None


def refresh_anthropic_oauth(stale_token: str | None = None, account: str | None = None) -> str | None:
    """Refresh the Anthropic OAuth access token using the stored refresh token."""
    account_key = account or _DEFAULT

    with _refresh_lock:
        oauth = _in_memory_oauth_cache.get(account_key)
        if oauth:
            current_tok = oauth.get("accessToken")
            if stale_token is not None:
                if isinstance(current_tok, str) and current_tok != stale_token:
                    _cached_key_cache[account_key] = current_tok
                    _cached_time_cache[account_key] = time.time()
                    return current_tok
            else:
                expires_at = oauth.get("expiresAt")
                if isinstance(expires_at, (int, float)):
                    if (expires_at / 1000.0) - 60 >= time.time():
                        if isinstance(current_tok, str) and current_tok:
                            _cached_key_cache[account_key] = current_tok
                            _cached_time_cache[account_key] = time.time()
                            return current_tok

        if account is None:
            # ---- Default account: legacy resolution (keychain + ~/.claude/.credentials.json) ----
            user = os.environ.get("USER") or ""
            if not user:
                return None

            # 1. Sourcing from Keychain with corruption check / cleanup
            keychain_exists = False
            keychain_bad = False
            keychain_raw = None
            keychain_data = None

            if sys.platform == "darwin" and os.environ.get("BYPASS_KEYCHAIN") != "1":
                try:
                    proc = subprocess.run(
                        ["security", "find-generic-password", "-s", _KEYCHAIN_SERVICE_CREDENTIALS, "-a", user, "-w"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if proc.returncode == 0:
                        keychain_exists = True
                        keychain_raw = proc.stdout.strip()
                except Exception:
                    pass

            if keychain_exists and keychain_raw:
                try:
                    parsed = json.loads(keychain_raw)
                    if isinstance(parsed, dict) and isinstance(parsed.get("claudeAiOauth"), dict) and isinstance(parsed["claudeAiOauth"].get("refreshToken"), str) and parsed["claudeAiOauth"]["refreshToken"].strip():
                        keychain_data = parsed
                    else:
                        keychain_bad = True
                except Exception:
                    keychain_bad = True

                if keychain_bad:
                    log.warning("macOS Keychain item exists but is bad. Deleting.")
                    try:
                        subprocess.run(
                            ["security", "delete-generic-password", "-s", _KEYCHAIN_SERVICE_CREDENTIALS, "-a", user],
                            capture_output=True, timeout=5
                        )
                    except Exception:
                        pass
                    keychain_exists = False

            # 2. Sourcing from Fallback Credentials File
            file_data = None
            path = Path.home() / ".claude" / ".credentials.json"
            if path.exists():
                try:
                    file_raw = path.read_text(encoding="utf-8").strip()
                    if file_raw:
                        parsed = json.loads(file_raw)
                        if isinstance(parsed, dict):
                            file_data = parsed
                except Exception:
                    pass

            # 3. Source refreshToken from memory, keychain, or file
            rt = None
            if _in_memory_oauth_cache.get(account_key) and isinstance(_in_memory_oauth_cache[account_key].get("refreshToken"), str) and _in_memory_oauth_cache[account_key]["refreshToken"].strip():
                rt = _in_memory_oauth_cache[account_key]["refreshToken"].strip()

            if not rt and keychain_data:
                claude_oauth = keychain_data.get("claudeAiOauth")
                if isinstance(claude_oauth, dict) and isinstance(claude_oauth.get("refreshToken"), str) and claude_oauth["refreshToken"].strip():
                    rt = claude_oauth["refreshToken"].strip()

            if not rt and file_data:
                claude_oauth = file_data.get("claudeAiOauth")
                if isinstance(claude_oauth, dict) and isinstance(claude_oauth.get("refreshToken"), str) and claude_oauth["refreshToken"].strip():
                    rt = claude_oauth["refreshToken"].strip()

            if not rt:
                return None

            # Build base dictionary for writing back
            if keychain_data:
                data = keychain_data
            elif file_data:
                data = file_data
            else:
                data = {}

            if "claudeAiOauth" not in data or not isinstance(data["claudeAiOauth"], dict):
                data["claudeAiOauth"] = {}
            claude_oauth = data["claudeAiOauth"]
        else:
            # ---- Named account: source from per-account secret file ----
            from open_llm_proxy import account_registry

            try:
                account_path = account_registry.resolve_secret_ref("claude-cli", account)
            except account_registry.AccountRegistryError:
                return None
            if not isinstance(account_path, Path) or not account_path.exists():
                return None
            try:
                data = json.loads(account_path.read_bytes())
            except Exception:
                return None
            if not isinstance(data, dict):
                return None
            claude_oauth_data = data.get("claudeAiOauth")
            if not isinstance(claude_oauth_data, dict):
                return None
            rt = claude_oauth_data.get("refreshToken")
            if not isinstance(rt, str) or not rt.strip():
                return None
            rt = rt.strip()

            if "claudeAiOauth" not in data or not isinstance(data["claudeAiOauth"], dict):
                data["claudeAiOauth"] = {}
            claude_oauth = data["claudeAiOauth"]

        try:
            import httpx
            resp = httpx.post(
                "https://console.anthropic.com/v1/oauth/token",
                headers={"Content-Type": "application/json"},
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": rt,
                    "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
                },
                timeout=15.0
            )
            if resp.status_code != 200:
                return None
            res_data = resp.json()
        except Exception:
            return None

        if not isinstance(res_data, dict):
            return None

        access_token = res_data.get("access_token")
        refresh_token = res_data.get("refresh_token")
        expires_in = res_data.get("expires_in")

        if not isinstance(access_token, str) or not access_token or \
           not isinstance(refresh_token, str) or not refresh_token or \
           not isinstance(expires_in, (int, float)):
            return None

        claude_oauth["accessToken"] = access_token
        claude_oauth["refreshToken"] = refresh_token
        now_ms = int(time.time() * 1000)
        expires_at_val = now_ms + int(expires_in * 1000)
        claude_oauth["expiresAt"] = expires_at_val

        updated_json = json.dumps(data)

        if account is None:
            # ---- Default account write-back (file + keychain) ----
            # 1. Primary write to the credentials file
            file_success = False
            try:
                path = Path.home() / ".claude" / ".credentials.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = path.with_suffix(".json.tmp")
                fd = os.open(str(tmp_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(updated_json)
                tmp_path.replace(path)

                if path.read_text(encoding="utf-8") == updated_json:
                    file_success = True
            except Exception as e:
                log.error("Failed to write primary credentials file: %s", e)

            if not file_success:
                raise RuntimeError("Primary credentials store write-back failed verification.")

            # 2. Best-effort Keychain write on macOS
            if sys.platform == "darwin" and os.environ.get("BYPASS_KEYCHAIN") != "1":
                keychain_verified_good = False
                try:
                    proc_write = subprocess.run(
                        ["security", "add-generic-password", "-U", "-s", _KEYCHAIN_SERVICE_CREDENTIALS, "-a", user, "-w"],
                        input=updated_json + "\n" + updated_json + "\n",
                        text=True, capture_output=True, timeout=10
                    )
                    proc_read = subprocess.run(
                        ["security", "find-generic-password", "-s", _KEYCHAIN_SERVICE_CREDENTIALS, "-a", user, "-w"],
                        capture_output=True, text=True, timeout=10
                    )
                    if proc_read.returncode == 0 and proc_read.stdout.strip() == updated_json:
                        keychain_verified_good = True
                except Exception as e:
                    log.warning("macOS Keychain write threw exception: %s", e)

                if not keychain_verified_good:
                    log.warning("macOS Keychain write is not verified-good. Deleting keychain item.")
                    try:
                        subprocess.run(
                            ["security", "delete-generic-password", "-s", _KEYCHAIN_SERVICE_CREDENTIALS, "-a", user],
                            capture_output=True, timeout=5
                        )
                        proc_verify = subprocess.run(
                            ["security", "find-generic-password", "-s", _KEYCHAIN_SERVICE_CREDENTIALS, "-a", user, "-w"],
                            capture_output=True, text=True, timeout=5
                        )
                        if proc_verify.returncode == 0:
                            stored_val = proc_verify.stdout.strip()
                            if stored_val != updated_json:
                                subprocess.run(
                                    ["security", "add-generic-password", "-U", "-s", _KEYCHAIN_SERVICE_CREDENTIALS, "-a", user, "-w"],
                                    input=updated_json + "\n" + updated_json + "\n",
                                    text=True, capture_output=True, timeout=10
                                )
                                proc_reverify = subprocess.run(
                                    ["security", "find-generic-password", "-s", _KEYCHAIN_SERVICE_CREDENTIALS, "-a", user, "-w"],
                                    capture_output=True, text=True, timeout=5
                                )
                                if proc_reverify.returncode != 0 or proc_reverify.stdout.strip() != updated_json:
                                    raise RuntimeError("Stale keychain item remains readable and could not be updated.")
                    except Exception as e:
                        if isinstance(e, RuntimeError):
                            raise
                        raise RuntimeError("Error occurred during keychain fallback/verification cleanup.") from e
        else:
            # ---- Named account write-back (per-account file only, no keychain) ----
            file_success = False
            try:
                account_path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_str = tempfile.mkstemp(dir=str(account_path.parent), prefix="creds_tmp_")
                tmp_path = Path(tmp_str)
                try:
                    os.chmod(fd, 0o600)
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        f.write(updated_json)
                    tmp_path.replace(account_path)
                except Exception:
                    if tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except Exception:
                            pass
                    raise
                if account_path.read_text(encoding="utf-8") == updated_json:
                    file_success = True
            except Exception as e:
                log.error("Failed to write account file: %s", e)

            if not file_success:
                raise RuntimeError("Account credentials store write-back failed verification.")

        _in_memory_oauth_cache[account_key] = {
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "expiresAt": expires_at_val
        }
        _cached_key_cache[account_key] = access_token
        _cached_time_cache[account_key] = time.time()
        return access_token


def _from_keychain() -> str | None:
    user = os.environ.get("USER") or ""
    if not user:
        return None

    if sys.platform == "darwin" and os.environ.get("BYPASS_KEYCHAIN") != "1":
        try:
            proc = subprocess.run(
                ["security", "find-generic-password", "-s", _KEYCHAIN_SERVICE_PRIMARY, "-a", user, "-w"],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0:
                out = proc.stdout.strip()
                if out:
                    return out
        except Exception:
            pass

        try:
            proc = subprocess.run(
                ["security", "find-generic-password", "-s", _KEYCHAIN_SERVICE_CREDENTIALS, "-a", user, "-w"],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0:
                out = proc.stdout.strip()
                if out:
                    try:
                        data = json.loads(out)
                        if isinstance(data, dict):
                            claude_oauth = data.get("claudeAiOauth")
                            if isinstance(claude_oauth, dict):
                                expires_at = claude_oauth.get("expiresAt")
                                if isinstance(expires_at, (int, float)):
                                    if (expires_at / 1000.0) - 60 < time.time():
                                        refreshed = refresh_anthropic_oauth()
                                        if refreshed:
                                            return refreshed
                                tok = claude_oauth.get("accessToken")
                                if tok and isinstance(tok, str):
                                    rt = claude_oauth.get("refreshToken")
                                    if isinstance(rt, str) and rt:
                                        _in_memory_oauth_cache[_DEFAULT] = {
                                            "accessToken": tok.strip(),
                                            "refreshToken": rt,
                                            "expiresAt": expires_at
                                        }
                                    return tok.strip()
                    except Exception:
                        pass
        except Exception:
            pass

        return None

    if sys.platform.startswith("linux"):
        for attrs in (
            ["service", "Claude Code", "account", user],
            ["account", "Claude Code"],
        ):
            try:
                proc = subprocess.run(
                    ["secret-tool", "lookup"] + attrs,
                    capture_output=True, text=True, timeout=5,
                )
                if proc.returncode == 0:
                    out = proc.stdout.strip()
                    if out:
                        return out
            except Exception:
                pass
        return None

    return None


def _from_claude_json() -> str | None:
    path = Path.home() / ".claude.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    for key in ("primaryApiKey", "api_key", "apiKey", "anthropic_api_key"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _from_credentials_file() -> str | None:
    path = Path.home() / ".claude" / ".credentials.json"
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return raw if raw.startswith("sk-") else None

    if isinstance(data, str):
        return data.strip() or None

    if isinstance(data, dict):
        claude_oauth = data.get("claudeAiOauth")
        if isinstance(claude_oauth, dict):
            expires_at = claude_oauth.get("expiresAt")
            if isinstance(expires_at, (int, float)):
                if (expires_at / 1000.0) - 60 < time.time():
                    refreshed = refresh_anthropic_oauth()
                    if refreshed:
                        return refreshed
            tok = claude_oauth.get("accessToken")
            if tok and isinstance(tok, str):
                rt = claude_oauth.get("refreshToken")
                if isinstance(rt, str) and rt:
                    _in_memory_oauth_cache[_DEFAULT] = {
                        "accessToken": tok.strip(),
                        "refreshToken": rt,
                        "expiresAt": expires_at
                    }
                return tok.strip()

        for key in ("api_key", "apiKey", "anthropic_api_key", "ANTHROPIC_API_KEY"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

        for v in data.values():
            if isinstance(v, dict):
                for key in ("api_key", "apiKey", "anthropic_api_key", "accessToken"):
                    inner = v.get(key)
                    if isinstance(inner, str) and inner.strip():
                        return inner.strip()
    return None


def _from_account_file(account: str) -> str | None:
    """Resolve credentials from a named account's per-account secret file."""
    from open_llm_proxy import account_registry

    try:
        resolved = account_registry.resolve_secret_ref("claude-cli", account)
    except account_registry.AccountRegistryError:
        return None
    if not isinstance(resolved, Path):
        return None
    path: Path = resolved
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
    except Exception:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return raw.decode("utf-8", errors="replace").strip() or None

    if isinstance(data, str):
        return data.strip() or None

    if isinstance(data, dict):
        claude_oauth = data.get("claudeAiOauth")
        if isinstance(claude_oauth, dict):
            expires_at = claude_oauth.get("expiresAt")
            if isinstance(expires_at, (int, float)):
                if (expires_at / 1000.0) - 60 < time.time():
                    refreshed = refresh_anthropic_oauth(account=account)
                    if refreshed:
                        return refreshed
            tok = claude_oauth.get("accessToken")
            if tok and isinstance(tok, str):
                rt = claude_oauth.get("refreshToken")
                if isinstance(rt, str) and rt:
                    _in_memory_oauth_cache[account] = {
                        "accessToken": tok.strip(),
                        "refreshToken": rt,
                        "expiresAt": expires_at,
                    }
                return tok.strip()

        for key in ("api_key", "apiKey", "anthropic_api_key", "ANTHROPIC_API_KEY"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

        for v in data.values():
            if isinstance(v, dict):
                for key in ("api_key", "apiKey", "anthropic_api_key", "accessToken"):
                    inner = v.get(key)
                    if isinstance(inner, str) and inner.strip():
                        return inner.strip()

    return None


def _resolve_active_claude_account() -> str | None:
    """Return the active claude-cli account name, or None for default/unset.

    Safely read the registry; any error yields None (legacy default path).
    """
    try:
        from open_llm_proxy import account_registry as _ar
        active = _ar.active_account("claude-cli")
        if active is not None and active != "default":
            return active
    except Exception:
        pass
    return None


def get_api_key(account: str | None = None) -> str:
    effective = account if account is not None else _resolve_active_claude_account()
    account_key = effective if effective is not None else _DEFAULT
    now = time.time()

    oauth = _in_memory_oauth_cache.get(account_key)
    if oauth:
        expires_at = oauth.get("expiresAt")
        if isinstance(expires_at, (int, float)):
            if (expires_at / 1000.0) - 60 < now:
                refreshed = refresh_anthropic_oauth(account=effective)
                if refreshed:
                    _cached_key_cache[account_key] = refreshed
                    _cached_time_cache[account_key] = now
                    return refreshed
        tok = oauth.get("accessToken")
        if isinstance(tok, str) and tok.strip():
            _cached_key_cache[account_key] = tok.strip()
            _cached_time_cache[account_key] = now
            return _cached_key_cache[account_key]

    if _cached_key_cache.get(account_key) and (now - _cached_time_cache.get(account_key, 0.0) < _TTL):
        return _cached_key_cache[account_key]

    if effective is not None:
        key = _from_account_file(effective)
        if key:
            _cached_key_cache[account_key] = key
            _cached_time_cache[account_key] = now
            return key
    else:
        for strategy in (_from_env, _from_keychain, _from_claude_json, _from_credentials_file):
            key = strategy()
            if key:
                _cached_key_cache[account_key] = key
                _cached_time_cache[account_key] = now
                return key

    raise RuntimeError("No Anthropic credentials found.")


def clear_cache(account: str | None = None) -> None:
    account_key = account or _DEFAULT
    _cached_key_cache.pop(account_key, None)
    _cached_time_cache.pop(account_key, None)


def reset_oauth_state(account: str | None = None) -> None:
    account_key = account or _DEFAULT
    _in_memory_oauth_cache.pop(account_key, None)
