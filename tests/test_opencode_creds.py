from __future__ import annotations

import json
import os
import pytest
from pathlib import Path
from open_llm_proxy.opencode_creds import (
    get_opencode_api_key,
    OpenCodeAuthError,
    _get_opencode_auth_path,
)


def test_get_opencode_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-opencode-env123")
    assert get_opencode_api_key() == "sk-opencode-env123"


def test_get_opencode_api_key_from_file(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    
    auth_dir = tmp_path / "opencode"
    auth_dir.mkdir()
    auth_file = auth_dir / "auth.json"
    
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(auth_file))
    
    # Valid auth json
    auth_file.write_text(
        json.dumps({
            "opencode": {
                "type": "api",
                "key": "sk-opencode-file456"
            }
        }),
        encoding="utf-8"
    )
    
    assert get_opencode_api_key() == "sk-opencode-file456"


def test_get_opencode_api_key_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-opencode-env-wins")
    
    auth_dir = tmp_path / "opencode"
    auth_dir.mkdir()
    auth_file = auth_dir / "auth.json"
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(auth_file))
    
    auth_file.write_text(
        json.dumps({
            "opencode": {
                "type": "api",
                "key": "sk-opencode-file-loses"
            }
        }),
        encoding="utf-8"
    )
    
    assert get_opencode_api_key() == "sk-opencode-env-wins"


def test_get_opencode_api_key_missing_file(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    
    non_existent = tmp_path / "nonexistent" / "auth.json"
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(non_existent))
    
    with pytest.raises(OpenCodeAuthError, match="No OpenCode API key found"):
        get_opencode_api_key()


def test_get_opencode_api_key_malformed_json(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    
    auth_file = tmp_path / "auth.json"
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(auth_file))
    
    auth_file.write_text("not-json", encoding="utf-8")
    
    with pytest.raises(OpenCodeAuthError, match="Malformed JSON"):
        get_opencode_api_key()


def test_get_opencode_api_key_not_dict(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    
    auth_file = tmp_path / "auth.json"
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(auth_file))
    
    auth_file.write_text(json.dumps(["not-a-dict"]), encoding="utf-8")
    
    with pytest.raises(OpenCodeAuthError, match="Expected a JSON object"):
        get_opencode_api_key()


def test_get_opencode_api_key_missing_opencode_sec(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    
    auth_file = tmp_path / "auth.json"
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(auth_file))
    
    auth_file.write_text(json.dumps({"github-copilot": {}}), encoding="utf-8")
    
    with pytest.raises(OpenCodeAuthError, match="Missing 'opencode' section"):
        get_opencode_api_key()


def test_get_opencode_api_key_missing_key_in_opencode(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    
    auth_file = tmp_path / "auth.json"
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(auth_file))
    
    auth_file.write_text(json.dumps({"opencode": {"type": "api"}}), encoding="utf-8")
    
    with pytest.raises(OpenCodeAuthError, match="Missing or invalid 'key'"):
        get_opencode_api_key()


def test_get_opencode_api_key_empty_key_in_opencode(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    
    auth_file = tmp_path / "auth.json"
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(auth_file))
    
    auth_file.write_text(json.dumps({"opencode": {"type": "api", "key": "   "}}), encoding="utf-8")
    
    with pytest.raises(OpenCodeAuthError, match="Missing or invalid 'key'"):
        get_opencode_api_key()


def test_get_opencode_api_key_unreadable_file(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    
    auth_file = tmp_path / "auth.json"
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(auth_file))
    
    auth_file.write_text("{}", encoding="utf-8")
    
    # Mock read_text to raise PermissionError
    def mock_read_text(*args, **kwargs):
        raise PermissionError("[Errno 13] Permission denied")
        
    monkeypatch.setattr(Path, "read_text", mock_read_text)
    
    with pytest.raises(OpenCodeAuthError) as exc_info:
        get_opencode_api_key()
        
    assert "Unreadable file" in str(exc_info.value)
    assert "PermissionError" in str(exc_info.value)
    assert "Permission denied" in str(exc_info.value)


def test_get_opencode_api_key_malformed_json_details(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    
    auth_file = tmp_path / "auth.json"
    monkeypatch.setenv("OPENCODE_AUTH_PATH", str(auth_file))
    
    # Use a malformed JSON string containing a dummy secret that shouldn't leak in the exception message
    auth_file.write_text('{"opencode": {"key": "secret-dont-leak"}', encoding="utf-8")
    
    with pytest.raises(OpenCodeAuthError) as exc_info:
        get_opencode_api_key()
        
    err_msg = str(exc_info.value)
    assert "Malformed JSON" in err_msg
    assert "JSONDecodeError" in err_msg
    assert "secret-dont-leak" not in err_msg
