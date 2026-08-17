import importlib

import pytest

import open_llm_proxy.config_paths as config_paths


def test_missing_env_config_does_not_fall_through(monkeypatch, tmp_path):
    fallback = tmp_path / "agent-config.yml"
    fallback.write_text("fallback: true")
    missing = tmp_path / "missing-agent-config.yml"
    monkeypatch.setenv("OPEN_LLM_PROXY_CONFIG", str(missing))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match=r"OPEN_LLM_PROXY_CONFIG.*missing-agent-config.yml"):
        config_paths.find_agent_config()


def test_existing_env_config_takes_precedence(monkeypatch, tmp_path):
    env_config = tmp_path / "env-agent-config.yml"
    cwd_config = tmp_path / "agent-config.yml"
    xdg_config_home = tmp_path / "empty-xdg"
    xdg_config_home.mkdir()
    env_config.write_text("env: true")
    cwd_config.write_text("cwd: true")
    monkeypatch.setenv("OPEN_LLM_PROXY_CONFIG", str(env_config))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_home))
    monkeypatch.chdir(tmp_path)

    assert config_paths.find_agent_config() == env_config


def test_existing_env_config_is_used_when_no_other_config_exists(monkeypatch, tmp_path):
    env_config = tmp_path / "env-agent-config.yml"
    xdg_config_home = tmp_path / "empty-xdg"
    xdg_config_home.mkdir()
    cwd = tmp_path / "missing-cwd"
    cwd.mkdir()
    env_config.write_text("env: true")
    monkeypatch.setenv("OPEN_LLM_PROXY_CONFIG", str(env_config))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_home))
    monkeypatch.chdir(cwd)

    assert config_paths.find_agent_config() == env_config


def test_empty_xdg_config_home_uses_default_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    monkeypatch.setattr(config_paths.Path, "home", lambda: tmp_path)

    assert config_paths.resolve_config_dir() == tmp_path / ".config" / "open-llm-proxy"


def test_config_dir_override_expands_user(monkeypatch, tmp_path):
    monkeypatch.setenv("OPEN_LLM_PROXY_CONFIG_DIR", "~/tildedir")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert config_paths.resolve_config_dir() == tmp_path / "tildedir"


def test_rate_limit_default_database_path_resolves_after_env_change(monkeypatch, tmp_path):
    import open_llm_proxy.rate_limit_state as rate_limit_state

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    monkeypatch.setenv("OPEN_LLM_PROXY_CONFIG_DIR", str(initial_dir))
    rate_limit_state = importlib.reload(rate_limit_state)
    monkeypatch.setenv("OPEN_LLM_PROXY_CONFIG_DIR", str(tmp_path))

    assert rate_limit_state.load_rate_limit_policy_from_data({})["database_path"] == (
        tmp_path / "state.sqlite3"
    )
