from __future__ import annotations

from pathlib import Path

import pytest

from open_llm_proxy.openrouter_creds import (
    get_api_key,
    get_persisted_api_key,
    save_api_key,
)


def test_get_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env123")
    assert get_api_key() == "sk-or-env123"


def test_get_api_key_from_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    env_dir = fake_home / ".config" / "open-llm-proxy"
    env_dir.mkdir(parents=True)
    env_file = env_dir / "env"
    env_file.write_text("OPENROUTER_API_KEY=sk-or-file123\n", encoding="utf-8")

    assert get_api_key() == "sk-or-file123"
    assert get_persisted_api_key() == "sk-or-file123"


def test_persisted_api_key_ignores_shell_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-shell-only")
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    with pytest.raises(RuntimeError, match="absent"):
        get_persisted_api_key()


def test_get_api_key_unreadable_file_fails_closed(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    env_dir = fake_home / ".config" / "open-llm-proxy"
    env_dir.mkdir(parents=True)
    env_file = env_dir / "env"
    env_file.write_text("OPENROUTER_API_KEY=secret\n", encoding="utf-8")

    def mock_read_text(*args, **kwargs):
        raise PermissionError("unreadable")

    monkeypatch.setattr(Path, "read_text", mock_read_text)

    with pytest.raises(PermissionError, match="unreadable"):
        get_api_key()


def test_save_api_key_unreadable_file_never_replaces(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    env_dir = fake_home / ".config" / "open-llm-proxy"
    env_dir.mkdir(parents=True)
    env_file = env_dir / "env"
    env_file.write_text("OPENROUTER_API_KEY=oldsecret\n", encoding="utf-8")

    def mock_read_text(*args, **kwargs):
        raise PermissionError("unreadable")

    monkeypatch.setattr(Path, "read_text", mock_read_text)

    with pytest.raises(PermissionError, match="unreadable"):
        save_api_key("newsecret")

    # Verify we didn't write anything new or delete the old file
    # We have to bypass the mocked read_text to inspect the file
    monkeypatch.undo()
    assert env_file.read_text(encoding="utf-8") == "OPENROUTER_API_KEY=oldsecret\n"
