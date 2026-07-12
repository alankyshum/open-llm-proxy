import os
import pytest
import urllib.error
import argparse
from unittest.mock import MagicMock, patch
from pathlib import Path

from open_llm_proxy import cli
from open_llm_proxy.server_launcher import launch_server


def test_serve_cli_options_parsing(monkeypatch):
    """Test that CLI parse options are forwarded properly to launch_server."""
    calls = []
    monkeypatch.setattr(
        "open_llm_proxy.server_launcher.launch_server",
        lambda *args, **kwargs: calls.append(kwargs),
    )

    # All options provided via CLI
    code = cli.main([
        "serve",
        "--host", "127.0.0.1",
        "--port", "9000",
        "--disable-admin-ui",
        "--database-url", "postgresql://localhost/db",
        "--master-key", "sk-custom-master",
        "--ui-username", "admin",
        "--ui-password", "secret",
    ])
    assert code == 0
    assert len(calls) == 1
    c = calls[0]
    assert c["host"] == "127.0.0.1"
    assert c["port"] == 9000
    assert c["disable_admin_ui"] is True
    assert c["database_url"] == "postgresql://localhost/db"
    assert c["master_key"] == "sk-custom-master"
    assert c["ui_username"] == "admin"
    assert c["ui_password"] == "secret"


def test_launch_server_ui_default_on_degrades_without_db(monkeypatch, tmp_path):
    """UI is ON by default but gracefully degrades (no crash) when no DB is configured."""
    config_path = tmp_path / "agent-config.yml"
    config_path.write_text("file_settings:\n  opencode:\n    model: open-llm-proxy/google/gemini-flash\n")

    monkeypatch.setattr("open_llm_proxy.server_launcher.setup_callbacks", lambda *args, **kwargs: None)
    monkeypatch.setattr("open_llm_proxy.server_launcher.generate_config_from_data", lambda *args, **kwargs: {})
    monkeypatch.setattr("open_llm_proxy.server_launcher.find_agent_config", lambda: config_path)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)

    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DISABLE_ADMIN_UI", raising=False)

    mock_app = MagicMock()
    mock_sys_modules = {"litellm.proxy.proxy_server": MagicMock(app=mock_app)}
    with patch.dict("sys.modules", mock_sys_modules):
        # Default call: UI requested implicitly, no DB -> must NOT raise, UI disabled.
        launch_server()
        assert os.environ.get("DISABLE_ADMIN_UI") == "True"


def test_launch_server_ui_autogenerates_master_key_with_db(monkeypatch, tmp_path):
    """When a DB is present and UI is on, a master key is auto-provisioned (no crash)."""
    config_path = tmp_path / "agent-config.yml"
    config_path.write_text("file_settings:\n  opencode:\n    model: open-llm-proxy/google/gemini-flash\n")

    monkeypatch.setattr("open_llm_proxy.server_launcher.setup_callbacks", lambda *args, **kwargs: None)
    monkeypatch.setattr("open_llm_proxy.server_launcher.generate_config_from_data", lambda *args, **kwargs: {})
    monkeypatch.setattr("open_llm_proxy.server_launcher.find_agent_config", lambda: config_path)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)

    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DISABLE_ADMIN_UI", raising=False)

    mock_app = MagicMock()
    mock_sys_modules = {"litellm.proxy.proxy_server": MagicMock(app=mock_app)}
    with patch.dict("sys.modules", mock_sys_modules):
        # DB present, no key -> UI stays enabled and a key is generated.
        launch_server(database_url="postgresql://localhost/db")
        assert os.environ.get("DISABLE_ADMIN_UI") is None
        assert os.environ["DATABASE_URL"] == "postgresql://localhost/db"
        assert os.environ["LITELLM_MASTER_KEY"].startswith("sk-")


def test_launch_server_allows_db_less_operation_when_ui_disabled(monkeypatch, tmp_path):
    """When UI is disabled, launch_server doesn't require master key or DB URL."""
    config_path = tmp_path / "agent-config.yml"
    config_path.write_text("file_settings:\n  opencode:\n    model: open-llm-proxy/google/gemini-flash\n")

    monkeypatch.setattr("open_llm_proxy.server_launcher.setup_callbacks", lambda *args, **kwargs: None)
    monkeypatch.setattr("open_llm_proxy.server_launcher.generate_config_from_data", lambda *args, **kwargs: {})
    monkeypatch.setattr("open_llm_proxy.server_launcher.find_agent_config", lambda: config_path)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)

    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    mock_app = MagicMock()
    mock_sys_modules = {"litellm.proxy.proxy_server": MagicMock(app=mock_app)}
    with patch.dict("sys.modules", mock_sys_modules):
        # UI disabled explicitly via argument
        launch_server(disable_admin_ui=True)
        assert os.environ.get("DISABLE_ADMIN_UI") == "True"


def test_launch_server_precedence_argument_over_env(monkeypatch, tmp_path):
    """Arguments should take precedence over existing environment variables."""
    config_path = tmp_path / "agent-config.yml"
    config_path.write_text("file_settings:\n  opencode:\n    model: open-llm-proxy/google/gemini-flash\n")

    monkeypatch.setattr("open_llm_proxy.server_launcher.setup_callbacks", lambda *args, **kwargs: None)
    monkeypatch.setattr("open_llm_proxy.server_launcher.generate_config_from_data", lambda *args, **kwargs: {})
    monkeypatch.setattr("open_llm_proxy.server_launcher.find_agent_config", lambda: config_path)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)

    # Set environment variables
    monkeypatch.setenv("LITELLM_MASTER_KEY", "env-key")
    monkeypatch.setenv("DATABASE_URL", "env-db")
    monkeypatch.setenv("UI_USERNAME", "env-user")
    monkeypatch.setenv("UI_PASSWORD", "env-pass")

    mock_app = MagicMock()
    mock_sys_modules = {"litellm.proxy.proxy_server": MagicMock(app=mock_app)}
    with patch.dict("sys.modules", mock_sys_modules):
        # Arguments are provided which should override env
        launch_server(
            disable_admin_ui=False,
            master_key="arg-key",
            database_url="arg-db",
            ui_username="arg-user",
            ui_password="arg-pass"
        )
        assert os.environ["LITELLM_MASTER_KEY"] == "arg-key"
        assert os.environ["DATABASE_URL"] == "arg-db"
        assert os.environ["UI_USERNAME"] == "arg-user"
        assert os.environ["UI_PASSWORD"] == "arg-pass"


def test_ui_command_success(monkeypatch, capsys):
    """Test 'ui' command when URL is responsive and returns 200 OK."""
    # Mock urllib.request.urlopen to return a responsive mock page
    mock_response = MagicMock()
    mock_response.getcode.return_value = 200
    mock_response.read.return_value = b"<html>Welcome to LiteLLM Admin UI</html>"
    
    mock_urlopen = MagicMock()
    mock_urlopen.return_value.__enter__.return_value = mock_response
    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    # Mock webbrowser.open
    webbrowser_calls = []
    monkeypatch.setattr("webbrowser.open", lambda url: webbrowser_calls.append(url))

    code = cli.main(["ui", "--url", "http://127.0.0.1:8765/ui", "--open"])
    assert code == 0
    assert "Success! LiteLLM Admin UI is running and responding with 200 OK." in capsys.readouterr().out
    assert webbrowser_calls == ["http://127.0.0.1:8765/ui"]


def test_ui_command_disabled_middleware(monkeypatch, capsys):
    """Test 'ui' command when middleware is blocking the Admin UI (returns Admin UI is disabled)."""
    mock_response = MagicMock()
    mock_response.getcode.return_value = 200
    mock_response.read.return_value = b"Admin UI is disabled."
    
    mock_urlopen = MagicMock()
    mock_urlopen.return_value.__enter__.return_value = mock_response
    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    code = cli.main(["ui", "--url", "http://127.0.0.1:8765/ui"])
    assert code == 1
    err_out = capsys.readouterr().err
    assert "Error: The server responded, but the Admin UI is disabled (middleware block)." in err_out


def test_ui_command_http_error(monkeypatch, capsys):
    """Test 'ui' command when HTTPError occurs (e.g., 404)."""
    # Mock HTTPError 404
    def mock_urlopen_error(*args, **kwargs):
        raise urllib.error.HTTPError("http://127.0.0.1:8765/ui", 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen_error)

    code = cli.main(["ui", "--url", "http://127.0.0.1:8765/ui"])
    assert code == 1
    err_out = capsys.readouterr().err
    assert "Error: Admin UI not found (404) at http://127.0.0.1:8765/ui. Is the UI disabled or port incorrect?" in err_out


def test_ui_command_unreachable(monkeypatch, capsys):
    """Test 'ui' command when server is down / unreachable."""
    def mock_urlopen_unreachable(*args, **kwargs):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen_unreachable)

    code = cli.main(["ui", "--url", "http://127.0.0.1:8765/ui"])
    assert code == 1
    err_out = capsys.readouterr().err
    assert "Is the open-llm-proxy server running?" in err_out
