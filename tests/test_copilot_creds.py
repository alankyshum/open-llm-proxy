from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from open_llm_proxy import copilot_creds


@pytest.mark.parametrize(
    "endpoint, expected",
    [
        ("not-a-url", False),
        ("https://", False),
        ("http://insecure.example", False),
        ("https://user:pw@host", False),
        ("https://valid.example", True),
    ],
)
def test_valid_endpoint_url(endpoint, expected):
    assert copilot_creds._valid_endpoint_url(endpoint) is expected


@pytest.mark.anyio
@pytest.mark.parametrize(
    "endpoint",
    ["not-a-url", "https://", "http://insecure.example", "https://user:pw@host"],
)
async def test_endpoint_discovery_invalid_endpoint_falls_back(
    monkeypatch,
    endpoint,
):
    monkeypatch.setenv("COPILOT_OAUTH_TOKEN", "ghu_endpoint_test")
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {"endpoints": {"api": endpoint}}
    with patch("httpx.AsyncClient.get", return_value=response):
        _, api_url = await copilot_creds.get_copilot_token()
    assert api_url == copilot_creds._DEFAULT_ENDPOINT


@pytest.fixture(autouse=True)
def clean_copilot_cache(monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("OPEN_LLM_PROXY_CONFIG_DIR", raising=False)
    monkeypatch.delenv("OLP_CONFIG_DIR", raising=False)
    copilot_creds.clear_oauth_cache()
    copilot_creds.invalidate_short_lived()
    yield
    copilot_creds.clear_oauth_cache()
    copilot_creds.invalidate_short_lived()


@pytest.fixture
def isolated_auth_path(monkeypatch, tmp_path):
    """Redirect _get_opencode_auth_path to tmp_path so tests don't touch real auth.json."""
    fake_path = tmp_path / "auth" / "auth.json"
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(copilot_creds, "_get_opencode_auth_path", lambda: fake_path)
    return fake_path


def test_get_oauth_token_from_env(monkeypatch):
    monkeypatch.setenv("COPILOT_OAUTH_TOKEN", "DUMMY-NOT-A-SECRET-env-token")
    assert copilot_creds.get_oauth_token() == "DUMMY-NOT-A-SECRET-env-token"


def test_get_oauth_token_from_opencode_auth(monkeypatch, tmp_path):
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    auth_json = fake_home / ".local" / "share" / "opencode" / "auth.json"
    auth_json.parent.mkdir(parents=True)
    auth_json.write_text(
        json.dumps({"github-copilot": {"refresh": "DUMMY-NOT-A-SECRET-refresh-token"}}),
        encoding="utf-8",
    )

    assert copilot_creds.get_oauth_token() == "DUMMY-NOT-A-SECRET-refresh-token"


def test_opencode_token_is_mirrored_to_fallback_file(monkeypatch, tmp_path):
    """A token read from OpenCode is persisted so it survives auth.json loss."""
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    auth_json = fake_home / ".local" / "share" / "opencode" / "auth.json"
    auth_json.parent.mkdir(parents=True)
    auth_json.write_text(
        json.dumps({"github-copilot": {"refresh": "ghu_mirrored123"}}), encoding="utf-8"
    )

    assert copilot_creds.get_oauth_token() == "ghu_mirrored123"

    fallback_file = fake_home / ".config" / "open-llm-proxy" / "copilot.json"
    assert fallback_file.is_file()
    assert json.loads(fallback_file.read_text())["oauth_token"] == "ghu_mirrored123"
    assert fallback_file.stat().st_mode & 0o777 == 0o600

    # OpenCode clears its auth.json; the mirrored copy keeps Copilot alive.
    auth_json.unlink()
    copilot_creds.clear_oauth_cache()
    assert copilot_creds.get_oauth_token() == "ghu_mirrored123"


def test_opencode_token_mirror_does_not_shadow_rotated_token(monkeypatch, tmp_path):
    """OpenCode stays authoritative; a stale mirror must never win."""
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    fallback_file = fake_home / ".config" / "open-llm-proxy" / "copilot.json"
    fallback_file.parent.mkdir(parents=True)
    fallback_file.write_text(json.dumps({"oauth_token": "ghu_stale"}), encoding="utf-8")

    auth_json = fake_home / ".local" / "share" / "opencode" / "auth.json"
    auth_json.parent.mkdir(parents=True)
    auth_json.write_text(
        json.dumps({"github-copilot": {"refresh": "ghu_rotated"}}), encoding="utf-8"
    )

    assert copilot_creds.get_oauth_token() == "ghu_rotated"
    assert json.loads(fallback_file.read_text())["oauth_token"] == "ghu_rotated"


def test_get_oauth_token_from_fallback_file(monkeypatch, tmp_path):
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    fallback_file = fake_home / ".config" / "open-llm-proxy" / "copilot.json"
    fallback_file.parent.mkdir(parents=True)
    fallback_file.write_text(json.dumps({"oauth_token": "ghu_fallback123"}), encoding="utf-8")

    assert copilot_creds.get_oauth_token() == "ghu_fallback123"


def test_get_oauth_token_keychain_darwin(monkeypatch):
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("USER", "testuser")
    monkeypatch.setattr(sys, "platform", "darwin")

    def mock_run(args, **kwargs):
        if "GitHub Copilot Proxy" in args:
            return MagicMock(returncode=0, stdout="ghu_keychain123\n")
        return MagicMock(returncode=1)

    monkeypatch.setattr(subprocess, "run", mock_run)
    assert copilot_creds.get_oauth_token() == "ghu_keychain123"


def test_get_oauth_token_secret_tool_linux(monkeypatch):
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("USER", "testuser")
    monkeypatch.setattr(sys, "platform", "linux")

    def mock_run(args, **kwargs):
        if "secret-tool" in args:
            return MagicMock(returncode=0, stdout="ghu_secrettool123\n")
        return MagicMock(returncode=1)

    monkeypatch.setattr(subprocess, "run", mock_run)
    # Mock shutil.which to return path to secret-tool
    monkeypatch.setattr(
        shutil, "which", lambda cmd: "/usr/bin/secret-tool" if cmd == "secret-tool" else None
    )

    assert copilot_creds.get_oauth_token() == "ghu_secrettool123"


import shutil  # staged imports preserve startup diagnostics  # noqa: E402


@pytest.mark.anyio
async def test_get_copilot_token_token_exchange(monkeypatch, isolated_auth_path):
    monkeypatch.setenv("COPILOT_OAUTH_TOKEN", "ghu_valid_oauth")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "token": "session_tok_123",
        "expires_at": 1900000000,
        "endpoints": {"api": "https://api.custom-copilot.com"},
    }

    # Patch httpx.AsyncClient.get
    with patch("httpx.AsyncClient.get", return_value=mock_response) as mock_get:
        token, api_url = await copilot_creds.get_copilot_token()
        assert token == "ghu_valid_oauth"
        assert api_url == "https://api.custom-copilot.com"
        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == "https://api.github.com/copilot_internal/user"
        headers_sent = mock_get.call_args[1]["headers"]
        assert headers_sent["Copilot-Integration-Id"] == "vscode-chat"
        assert headers_sent["Authorization"] == "Bearer ghu_valid_oauth"
        assert "X-Request-Id" in headers_sent


@pytest.mark.anyio
async def test_get_copilot_token_single_flight(monkeypatch):
    monkeypatch.setenv("COPILOT_OAUTH_TOKEN", "ghu_concurrent")
    monkeypatch.setattr(copilot_creds, "_read_opencode_auth_data", lambda: None)
    calls = 0

    async def fetch(oauth):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return copilot_creds._ShortLived(
            token=oauth,
            expires_at=int(time.time()) + 3600,
            endpoints_api="https://api.enterprise.githubcopilot.com",
        )

    monkeypatch.setattr(copilot_creds, "_fetch_short_lived", fetch)
    results = await asyncio.gather(*(copilot_creds.get_copilot_token() for _ in range(10)))
    assert calls == 1
    assert all(result == results[0] for result in results)


@pytest.mark.anyio
async def test_get_copilot_token_unexpired_access(monkeypatch, isolated_auth_path):
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    fake_data = {
        "access": "session_tok_from_opencode",
        "expires": int(time.time() + 1000),
    }
    with patch("open_llm_proxy.copilot_creds._read_opencode_auth_data", return_value=fake_data):
        token, api_url = await copilot_creds.get_copilot_token()
        assert token == "session_tok_from_opencode"
        assert api_url == "https://api.githubcopilot.com"


@pytest.mark.anyio
async def test_get_copilot_token_expired_access_goes_to_exchange(monkeypatch, isolated_auth_path):
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    fake_data = {
        "refresh": "gho_refresh_oauth",
        "access": "session_tok_expired",
        "expires": int(time.time() - 1000),
    }

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"endpoints": {"api": "https://api.githubcopilot.com"}}

    with (
        patch("open_llm_proxy.copilot_creds._read_opencode_auth_data", return_value=fake_data),
        patch("httpx.AsyncClient.get", return_value=mock_response) as mock_get,
    ):
        token, api_url = await copilot_creds.get_copilot_token()
        assert token == "gho_refresh_oauth"
        mock_get.assert_called_once()
        # Verify the gho_refresh_oauth token was used in Authorization header during exchange
        headers_sent = mock_get.call_args[1]["headers"]
        assert headers_sent["Authorization"] == "Bearer gho_refresh_oauth"
        assert headers_sent["Copilot-Integration-Id"] == "vscode-chat"
        assert "X-Request-Id" in headers_sent


@pytest.mark.anyio
async def test_get_copilot_token_404_falls_back_to_direct(monkeypatch, isolated_auth_path):
    """Token exchange 404 falls back to using raw OAuth token directly (no raise)."""
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    fake_data = {
        "refresh": "gho_no_copilot_scope",
        "access": "gho_no_copilot_scope",
        "expires": 0,
    }
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 404
    mock_response.text = '{"message":"Not Found"}'

    with (
        patch("open_llm_proxy.copilot_creds._read_opencode_auth_data", return_value=fake_data),
        patch("httpx.AsyncClient.get", return_value=mock_response) as mock_get,
    ):
        token, api_url = await copilot_creds.get_copilot_token()
        assert token == "gho_no_copilot_scope"  # falls back to raw token
        assert api_url == "https://api.githubcopilot.com"
        mock_get.assert_called_once()


@pytest.mark.anyio
async def test_get_copilot_token_writes_back_to_auth_json(monkeypatch, isolated_auth_path):
    """After successful exchange, write session token back to auth.json."""
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    original = {
        "github-copilot": {
            "type": "oauth",
            "refresh": "gho_valid_refresh",
            "access": "gho_valid_refresh",
            "expires": 0,
        }
    }
    isolated_auth_path.write_text(json.dumps(original), encoding="utf-8")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    int(time.time()) + 3600
    mock_response.json.return_value = {
        "endpoints": {"api": "https://api.githubcopilot.com"},
    }

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        token, _ = await copilot_creds.get_copilot_token()
        assert token == "gho_valid_refresh"

    updated = json.loads(isolated_auth_path.read_text(encoding="utf-8"))
    cp = updated["github-copilot"]
    assert cp["access"] == "gho_valid_refresh"
    assert cp["expires"] > int(time.time())
    assert cp["refresh"] == "gho_valid_refresh"
    assert cp["type"] == "oauth"
    assert not isolated_auth_path.with_suffix(".json.tmp").exists()


@pytest.mark.anyio
async def test_get_copilot_token_degenerate_state(monkeypatch, isolated_auth_path):
    """access==refresh (both gho_) with expires=0: exchange fails → falls back to direct token."""
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    fake_data = {
        "refresh": "gho_degenerate",
        "access": "gho_degenerate",
        "expires": 0,
    }
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 404
    mock_response.text = '{"message":"Not Found"}'

    with (
        patch("open_llm_proxy.copilot_creds._read_opencode_auth_data", return_value=fake_data),
        patch("httpx.AsyncClient.get", return_value=mock_response),
    ):
        token, api_url = await copilot_creds.get_copilot_token()
        assert token == "gho_degenerate"
        assert api_url == "https://api.githubcopilot.com"


@pytest.mark.anyio
async def test_get_copilot_token_no_token_raises(monkeypatch, isolated_auth_path):
    """No token at all → CopilotAuthError."""
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(copilot_creds, "_read_opencode_auth_data", lambda: None)
    monkeypatch.setattr(copilot_creds, "_read_fallback_file", lambda: None)
    monkeypatch.setattr(copilot_creds, "_read_keychain_macos", lambda: None)
    monkeypatch.setattr(copilot_creds, "_read_secret_tool_linux", lambda: None)
    with pytest.raises(copilot_creds.CopilotAuthError, match="No Copilot OAuth token found"):
        await copilot_creds.get_copilot_token()
