from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import time

import pytest
import httpx

from open_llm_proxy import copilot_creds


@pytest.fixture(autouse=True)
def clean_copilot_cache():
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
    monkeypatch.setenv("COPILOT_OAUTH_TOKEN", "ghu_env123")
    assert copilot_creds.get_oauth_token() == "ghu_env123"


def test_get_oauth_token_from_opencode_auth(monkeypatch, tmp_path):
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    auth_json = fake_home / ".local" / "share" / "opencode" / "auth.json"
    auth_json.parent.mkdir(parents=True)
    auth_json.write_text(json.dumps({
        "github-copilot": {"refresh": "ghu_opencode123"}
    }), encoding="utf-8")

    assert copilot_creds.get_oauth_token() == "ghu_opencode123"


def test_get_oauth_token_from_fallback_file(monkeypatch, tmp_path):
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    fallback_file = fake_home / ".config" / "kilo-claude-proxy" / "copilot.json"
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
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/secret-tool" if cmd == "secret-tool" else None)

    assert copilot_creds.get_oauth_token() == "ghu_secrettool123"


import shutil


@pytest.mark.anyio
async def test_get_copilot_token_token_exchange(monkeypatch, isolated_auth_path):
    monkeypatch.setenv("COPILOT_OAUTH_TOKEN", "ghu_valid_oauth")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "token": "session_tok_123",
        "expires_at": 1900000000,
        "endpoints": {"api": "https://api.custom-copilot.com"}
    }

    # Patch httpx.AsyncClient.get
    with patch("httpx.AsyncClient.get", return_value=mock_response) as mock_get:
        token, api_url = await copilot_creds.get_copilot_token()
        assert token == "session_tok_123"
        assert api_url == "https://api.custom-copilot.com"
        mock_get.assert_called_once()
        headers_sent = mock_get.call_args[1]["headers"]
        assert headers_sent["Copilot-Integration-Id"] == "vscode-chat"
        assert headers_sent["Authorization"] == "Bearer ghu_valid_oauth"
        assert "X-Request-Id" in headers_sent

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
    mock_response.json.return_value = {
        "token": "session_tok_exchanged",
        "expires_at": int(time.time() + 3600),
        "endpoints": {"api": "https://api.githubcopilot.com"}
    }

    with patch("open_llm_proxy.copilot_creds._read_opencode_auth_data", return_value=fake_data), \
         patch("httpx.AsyncClient.get", return_value=mock_response) as mock_get:
        token, api_url = await copilot_creds.get_copilot_token()
        assert token == "session_tok_exchanged"
        mock_get.assert_called_once()
        # Verify the gho_refresh_oauth token was used in Authorization header during exchange
        headers_sent = mock_get.call_args[1]["headers"]
        assert headers_sent["Authorization"] == "Bearer gho_refresh_oauth"
        assert headers_sent["Copilot-Integration-Id"] == "vscode-chat"
        assert "X-Request-Id" in headers_sent


@pytest.mark.anyio
async def test_get_copilot_token_404_raises_clear_error(monkeypatch, isolated_auth_path):
    """Token exchange 404 raises CopilotAuthError with scope hint."""
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    fake_data = {
        "refresh": "gho_no_copilot_scope",
        "access": "gho_no_copilot_scope",
        "expires": 0,
    }
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 404
    mock_response.text = '{"message":"Not Found"}'

    with patch("open_llm_proxy.copilot_creds._read_opencode_auth_data", return_value=fake_data), \
         patch("httpx.AsyncClient.get", return_value=mock_response) as mock_get:
        with pytest.raises(copilot_creds.CopilotAuthError) as exc_info:
            await copilot_creds.get_copilot_token()
        assert "copilot" in str(exc_info.value).lower()
        assert "scope" in str(exc_info.value).lower()
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
    future_ts = int(time.time()) + 3600
    mock_response.json.return_value = {
        "token": "session_tok_persisted",
        "expires_at": future_ts,
        "endpoints": {"api": "https://api.githubcopilot.com"},
    }

    with patch("httpx.AsyncClient.get", return_value=mock_response) as mock_get:
        token, _ = await copilot_creds.get_copilot_token()
        assert token == "session_tok_persisted"

    updated = json.loads(isolated_auth_path.read_text(encoding="utf-8"))
    cp = updated["github-copilot"]
    assert cp["access"] == "session_tok_persisted"
    assert cp["expires"] == future_ts
    assert cp["refresh"] == "gho_valid_refresh"
    assert cp["type"] == "oauth"
    assert not isolated_auth_path.with_suffix(".json.tmp").exists()


@pytest.mark.anyio
async def test_get_copilot_token_degenerate_state(monkeypatch, isolated_auth_path):
    """access==refresh (both gho_) with expires=0: error still propagates clearly."""
    monkeypatch.delenv("COPILOT_OAUTH_TOKEN", raising=False)
    fake_data = {
        "refresh": "gho_degenerate",
        "access": "gho_degenerate",
        "expires": 0,
    }
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 404
    mock_response.text = '{"message":"Not Found"}'

    with patch("open_llm_proxy.copilot_creds._read_opencode_auth_data", return_value=fake_data), \
         patch("httpx.AsyncClient.get", return_value=mock_response):
        with pytest.raises(copilot_creds.CopilotAuthError):
            await copilot_creds.get_copilot_token()
