from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from open_llm_proxy import creds


@pytest.fixture(autouse=True)
def clean_cache():
    creds.clear_cache()
    creds.reset_oauth_state()
    yield
    creds.clear_cache()
    creds.reset_oauth_state()


def test_get_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env123")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert creds.get_api_key() == "sk-ant-env123"


def test_get_api_key_from_env_auth_token(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-ant-token123")
    assert creds.get_api_key() == "sk-ant-token123"


def test_get_api_key_from_claude_json(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    claude_json = fake_home / ".claude.json"
    claude_json.write_text(json.dumps({"primaryApiKey": "sk-ant-json123"}), encoding="utf-8")

    assert creds.get_api_key() == "sk-ant-json123"


def test_get_api_key_from_credentials_file_raw(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    credentials_dir = fake_home / ".claude"
    credentials_dir.mkdir()
    credentials_json = credentials_dir / ".credentials.json"
    credentials_json.write_text("sk-ant-raw123", encoding="utf-8")

    assert creds.get_api_key() == "sk-ant-raw123"


def test_get_api_key_from_credentials_file_json(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    credentials_dir = fake_home / ".claude"
    credentials_dir.mkdir()
    credentials_json = credentials_dir / ".credentials.json"
    credentials_json.write_text(
        json.dumps({"api_key": "sk-ant-credentials-json"}), encoding="utf-8"
    )

    assert creds.get_api_key() == "sk-ant-credentials-json"


def test_get_api_key_from_keychain_darwin(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("USER", "testuser")
    monkeypatch.setattr(sys, "platform", "darwin")

    def mock_run(args, **kwargs):
        if "Claude Code" in args:
            return MagicMock(returncode=0, stdout="sk-ant-keychain123\n")
        return MagicMock(returncode=1)

    monkeypatch.setattr(subprocess, "run", mock_run)
    assert creds.get_api_key() == "sk-ant-keychain123"


def test_get_api_key_from_secret_tool_linux(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("USER", "testuser")
    monkeypatch.setattr(sys, "platform", "linux")

    def mock_run(args, **kwargs):
        if "secret-tool" in args:
            return MagicMock(returncode=0, stdout="sk-ant-secrettool123\n")
        return MagicMock(returncode=1)

    monkeypatch.setattr(subprocess, "run", mock_run)
    assert creds.get_api_key() == "sk-ant-secrettool123"


def test_local_machine_resolution_graceful():
    # If the local developer machine has credentials, this should resolve them successfully.
    # If not, it should raise RuntimeError cleanly, which we catch.
    try:
        key = creds.get_api_key()
        assert isinstance(key, str)
        assert len(key) > 0
    except RuntimeError as e:
        assert "No Anthropic credentials found." in str(e)
