from __future__ import annotations

import json
import asyncio
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import httpx

log = logging.getLogger("open_llm_proxy.copilot_creds")

COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
_KEYCHAIN_SERVICE = "GitHub Copilot Proxy"
def get_fallback_path() -> Path:
    return Path.home() / ".config" / "open-llm-proxy" / "copilot.json"


_COPILOT_USER_URL = "https://api.github.com/copilot_internal/user"
_DEFAULT_ENDPOINT = "https://api.githubcopilot.com"

EDITOR_VERSION = "vscode/1.95.3"
EDITOR_PLUGIN_VERSION = "copilot-chat/0.26.7"
USER_AGENT = "GitHubCopilotChat/0.26.7"
COPILOT_INTEGRATION_ID = "vscode-chat"
X_GITHUB_API_VERSION = "2025-04-01"


class CopilotAuthError(RuntimeError):
    """No usable Copilot credential available."""


_oauth_token_cache: Optional[str] = None
_oauth_token_lock = threading.Lock()


def clear_oauth_cache() -> None:
    global _oauth_token_cache
    with _oauth_token_lock:
        _oauth_token_cache = None


def _read_keychain_macos() -> Optional[str]:
    try:
        user = os.environ.get("USER") or ""
        out = subprocess.run(
            ["security", "find-generic-password",
             "-s", _KEYCHAIN_SERVICE, "-a", user, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            tok = out.stdout.strip()
            return tok or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return None


def _read_secret_tool_linux() -> Optional[str]:
    if not shutil.which("secret-tool"):
        return None
    try:
        user = os.environ.get("USER") or ""
        out = subprocess.run(
            ["secret-tool", "lookup",
             "service", _KEYCHAIN_SERVICE, "account", user],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            tok = out.stdout.strip()
            return tok or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return None


def _read_fallback_file() -> Optional[str]:
    path = get_fallback_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    tok = data.get("oauth_token") if isinstance(data, dict) else None
    return tok if isinstance(tok, str) and tok else None


def _write_fallback_file(token: str) -> None:
    """Mirror *token* into the last-resort fallback file (mode 0600).

    OpenCode owns ``auth.json`` and may rotate or clear it at any time. Without
    a mirrored copy that leaves no Copilot credential at all on the next
    process start, because a live proxy only keeps the token in memory.
    """
    path = get_fallback_path()
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        else:
            data = {}
    except (OSError, json.JSONDecodeError):
        data = {}
    if data.get("oauth_token") == token:
        return
    data["oauth_token"] = token
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.chmod(0o600)
        tmp.replace(path)
    except OSError:
        log.warning("copilot: failed to mirror OAuth token to %s", path)


def _get_opencode_auth_path() -> Path:
    path_str = os.environ.get("OPENCODE_AUTH_PATH")
    if path_str:
        return Path(path_str)
    return Path.home() / ".local" / "share" / "opencode" / "auth.json"


def _read_opencode_auth_data() -> Optional[dict]:
    path = _get_opencode_auth_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        copilot = data.get("github-copilot")
        if isinstance(copilot, dict):
            return copilot
    except Exception:
        pass
    return None


def _read_opencode_auth() -> Optional[str]:
    copilot = _read_opencode_auth_data()
    if not copilot:
        return None
    refresh = copilot.get("refresh")
    if isinstance(refresh, str) and refresh:
        return refresh
    access = copilot.get("access")
    if isinstance(access, str) and access:
        return access
    return None


def get_oauth_token() -> str:
    global _oauth_token_cache
    if _oauth_token_cache:
        return _oauth_token_cache
    with _oauth_token_lock:
        if _oauth_token_cache:
            return _oauth_token_cache

        env = os.environ.get("COPILOT_OAUTH_TOKEN")
        if env:
            _oauth_token_cache = env
            return env

        if sys.platform == "darwin" and os.environ.get("BYPASS_KEYCHAIN") != "1":
            tok = _read_keychain_macos()
            if tok:
                _oauth_token_cache = tok
                return tok
        else:
            tok = _read_secret_tool_linux()
            if tok:
                _oauth_token_cache = tok
                return tok

        tok = _read_opencode_auth()
        if tok:
            _oauth_token_cache = tok
            # Keep a durable copy; auth.json is owned by OpenCode.
            _write_fallback_file(tok)
            return tok

        tok = _read_fallback_file()
        if tok:
            _oauth_token_cache = tok
            return tok

        raise CopilotAuthError(
            "No Copilot OAuth token found."
        )


def _write_opencode_auth_back(copilot_entry: dict) -> None:
    """Atomically write the github-copilot entry back to auth.json (mode 0600)."""
    path = _get_opencode_auth_path()
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data["github-copilot"] = copilot_entry
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.chmod(0o600)
        tmp.rename(path)
    except Exception:
        log.warning("copilot: failed to write refreshed token back to auth.json")


@dataclass
class _ShortLived:
    token: str
    expires_at: int
    endpoints_api: str


_short_lived: Optional[_ShortLived] = None
_short_lived_lock = asyncio.Lock()


def invalidate_short_lived() -> None:
    global _short_lived
    _short_lived = None


def _exchange_token_headers(oauth: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {oauth}",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
        "Editor-Version": EDITOR_VERSION,
        "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
        "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
        "X-GitHub-Api-Version": X_GITHUB_API_VERSION,
        "X-Request-Id": str(uuid.uuid4()),
    }


def _valid_endpoint_url(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme == "https"
            and parsed.netloc != ""
            and parsed.hostname is not None
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        return False


async def _fetch_short_lived(oauth: str) -> _ShortLived:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(
            _COPILOT_USER_URL, headers=_exchange_token_headers(oauth),
        )
    if r.status_code == 401:
        clear_oauth_cache()
        raise CopilotAuthError(
            "GitHub returned 401 from /copilot_internal/user — stored OAuth token is invalid or revoked."
        )
    if r.status_code == 404:
        raise CopilotAuthError(
            "GitHub returned 404 from /copilot_internal/user — stored OAuth token lacks the required 'copilot' scope. "
            "Regenerate the token by running a Copilot task in opencode or re-authenticating."
        )
    if r.status_code != 200:
        raise CopilotAuthError(
            f"Copilot endpoint discovery failed: {r.status_code}"
        )
    data = r.json()
    endpoints = data.get("endpoints") or {}
    api = endpoints.get("api") if isinstance(endpoints, dict) else None
    if not _valid_endpoint_url(api):
        api = _DEFAULT_ENDPOINT
    # OpenCode-issued gho_/ghu_ tokens are accepted directly by the Copilot
    # endpoint; the VS Code token-exchange endpoint rejects them with 403.
    return _ShortLived(token=oauth, expires_at=int(time.time()) + 3600, endpoints_api=api)


async def get_copilot_token() -> tuple[str, str]:
    """Return (session_token, base_url), refreshing if within 60 s of expiry."""
    global _short_lived
    now = int(time.time())
    if _short_lived is not None and (_short_lived.expires_at - now) > 60:
        return _short_lived.token, _short_lived.endpoints_api

    # Hold the async lock across discovery and all cache population. This is
    # deliberately single-flight: startup callers must not stampede GitHub.
    async with _short_lived_lock:
        now = int(time.time())
        if _short_lived is not None and (_short_lived.expires_at - now) > 60:
            return _short_lived.token, _short_lived.endpoints_api
        return await _get_copilot_token_locked(now)


async def _get_copilot_token_locked(now: int) -> tuple[str, str]:
    global _short_lived
    copilot_data = _read_opencode_auth_data()
    if copilot_data:
        access = copilot_data.get("access")
        expires_val = copilot_data.get("expires")
        if isinstance(access, str) and access and expires_val is not None:
            try:
                expires_val = float(expires_val)
                if expires_val > 99999999999:
                    expires_sec = expires_val / 1000.0
                else:
                    expires_sec = expires_val

                # Skip degenerate/zero expires (was never properly set)
                if expires_sec <= 0:
                    log.debug("copilot: access token expires == 0 (unset), skipping")
                # Skip gho_/ghu_ tokens — they are OAuth tokens, not session tokens
                elif access.startswith("gho_") or access.startswith("ghu_"):
                    log.debug(
                        "copilot: access token starts with gho_/ghu_ (OAuth token, not session), skipping"
                    )
                # Allow small negative skew (up to 30s) to handle clock differences
                elif (expires_sec - now) > -30:
                    fresh = _ShortLived(
                        token=access,
                        expires_at=int(expires_sec),
                        endpoints_api=_DEFAULT_ENDPOINT,
                    )
                    _short_lived = fresh
                    log.info(
                        "copilot: using unexpired access token from opencode auth (expires_at=%d)",
                        fresh.expires_at,
                    )
                    return fresh.token, fresh.endpoints_api
                else:
                    log.debug(
                        "copilot: access token expired (%d < %d)",
                        int(expires_sec), int(now),
                    )
            except Exception:
                log.debug("Failed parsing opencode auth expires")

    oauth = get_oauth_token()
    if not oauth.startswith("gho_") and not oauth.startswith("ghu_"):
        fresh = _ShortLived(
            token=oauth,
            expires_at=now + 3600,
            endpoints_api=_DEFAULT_ENDPOINT,
        )
    else:
        try:
            fresh = await _fetch_short_lived(oauth)
            # Persist the exchanged session token back to auth.json
            if copilot_data is not None:
                copilot_data["access"] = fresh.token
                copilot_data["expires"] = fresh.expires_at
                _write_opencode_auth_back(copilot_data)
        except Exception as e:
            # Discovery failed. Fall back to using the raw OAuth token directly
            # against the default Copilot endpoint — GitHub Copilot
            # endpoints accept raw gho_/ghu_ tokens directly.
            if isinstance(e, CopilotAuthError):
                log.warning("copilot: token exchange failed, falling back to direct OAuth token")
            else:
                log.warning("copilot: token exchange error, falling back to direct OAuth token")
            fresh = _ShortLived(
                token=oauth,
                expires_at=now + 3600,
                endpoints_api=_DEFAULT_ENDPOINT,
            )

    _short_lived = fresh
    log.info(
        "copilot: refreshed short-lived token (expires_at=%d, api=%s)",
        fresh.expires_at, fresh.endpoints_api,
    )
    return fresh.token, fresh.endpoints_api
