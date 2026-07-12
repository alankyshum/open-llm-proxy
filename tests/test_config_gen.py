import pytest
from pathlib import Path
from open_llm_proxy.config_gen import (
    configured_model_tokens,
    parse_fallback_chain,
    map_token_to_deployment_params,
    generate_config,
)


def test_configured_model_tokens(tmp_path):
    config_file = tmp_path / "agent-config.yml"
    config_file.write_text(
        """
file_settings:
  opencode:
    model: "open-llm-proxy/[claude-cli/claude-sonnet-5,google/gemini-3.5-flash]"
agents:
  reviewer:
    model: "openrouter/z-ai/glm-5.2"
"""
    )

    assert configured_model_tokens(config_file) == {
        "claude-cli/claude-sonnet-5",
        "google/gemini-3.5-flash",
        "openrouter/z-ai/glm-5.2",
    }

def test_parse_fallback_chain_success():
    # dual prefix acceptance -> no longer dual prefix, open-llm-proxy only
    c1 = "open-llm-proxy/[claude-cli/claude-sonnet-5,github-copilot/claude-sonnet-5,openrouter/z-ai/glm-5.2]"
    tokens1 = parse_fallback_chain(c1)
    assert tokens1 == ["claude-cli/claude-sonnet-5", "github-copilot/claude-sonnet-5", "openrouter/z-ai/glm-5.2"]

    c2 = "open-llm-proxy/[claude-cli/claude-sonnet-5,github-copilot/claude-sonnet-5]"
    tokens2 = parse_fallback_chain(c2)
    assert tokens2 == ["claude-cli/claude-sonnet-5", "github-copilot/claude-sonnet-5"]

    # no prefix
    c3 = "[google/gemini-2.5-flash,github-copilot/gemini-2.5-pro]"
    tokens3 = parse_fallback_chain(c3)
    assert tokens3 == ["google/gemini-2.5-flash", "github-copilot/gemini-2.5-pro"]

    # plain model
    c4 = "github-copilot/gpt-5-mini"
    tokens4 = parse_fallback_chain(c4)
    assert tokens4 == ["github-copilot/gpt-5-mini"]

def test_parse_fallback_chain_failures():
    with pytest.raises(ValueError):
        parse_fallback_chain("[claude-cli/claude-sonnet-5") # unclosed
    with pytest.raises(ValueError):
        parse_fallback_chain("[]") # empty
    with pytest.raises(ValueError):
        parse_fallback_chain("[claude-cli]") # missing slash

def test_map_token_to_deployment_params(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-opencode-mock-key")
    # google gemini prefix mapping
    params_google = map_token_to_deployment_params("google/models/gemini-2.5-flash")
    assert params_google["model"] == "gemini/gemini-2.5-flash"
    assert params_google["api_key"] == "os.environ/GEMINI_API_KEY"

    params_google_bare = map_token_to_deployment_params("google/gemini-2.5-flash")
    assert params_google_bare["model"] == "gemini/gemini-2.5-flash"

    # opencode mapping
    params_opencode = map_token_to_deployment_params("opencode/big-pickle")
    assert params_opencode["model"] == "openai/big-pickle"
    assert params_opencode["api_base"] == "https://opencode.ai/zen/v1"
    assert params_opencode["api_key"] == "os.environ/OPENCODE_API_KEY"

    # claude-cli mapping
    params_claude = map_token_to_deployment_params("claude-cli/claude-sonnet-5")
    assert params_claude["model"] == "claude-cli/claude-sonnet-5"

    # copilot mapping
    params_copilot = map_token_to_deployment_params("github-copilot/claude-sonnet-5")
    assert params_copilot["model"] == "github-copilot/gh-claude-sonnet-5"
    assert params_copilot["api_key"] == "not-needed"

    # ollama-local mapping — routes to litellm ollama_chat provider against local daemon
    params_ollama = map_token_to_deployment_params("ollama-local/qwen2.5vl:7b")
    assert params_ollama["model"] == "ollama_chat/qwen2.5vl:7b"
    assert params_ollama["api_base"] == "http://127.0.0.1:11434"
    assert params_ollama["num_ctx"] == 16384
    assert params_ollama["num_predict"] == 4000
    assert params_ollama["think"] is False
    assert "api_key" not in params_ollama

    monkeypatch.setattr(
        "open_llm_proxy.openrouter_creds.get_persisted_api_key",
        lambda: "sk-or-persisted",
    )
    params_openrouter = map_token_to_deployment_params(
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
    )
    assert params_openrouter["model"] == "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
    assert params_openrouter["api_key"] == "os.environ/OPENROUTER_API_KEY"
    assert __import__("os").environ["OPENROUTER_API_KEY"] == "sk-or-persisted"

def test_generate_config_real(tmp_path):
    # Create a dummy agent-config.yml
    dummy_yaml = """
file_settings:
  opencode:
    model: "open-llm-proxy/[claude-cli/claude-sonnet-5,github-copilot/claude-sonnet-5,openrouter/z-ai/glm-5.2]"
    small_model: "github-copilot/gpt-5-mini"

agents:
  lead: {
    model: "github-copilot/claude-opus-4.8"
  }
  code-reviewer: {
    model: "github-copilot/gpt-5.5"
  }
"""
    config_file = tmp_path / "agent-config.yml"
    config_file.write_text(dummy_yaml)

    config_dict = generate_config(str(config_file))
    
    assert "model_list" in config_dict
    assert "fallbacks" not in config_dict
    assert "litellm_settings" in config_dict
    assert "router_settings" in config_dict

    model_list = config_dict["model_list"]
    litellm_settings = config_dict["litellm_settings"]
    router_settings = config_dict["router_settings"]

    assert "fallbacks" in litellm_settings
    assert litellm_settings["drop_params"] is True
    assert "fallbacks" in router_settings
    fallbacks_l = litellm_settings["fallbacks"]
    fallbacks_r = router_settings["fallbacks"]
    assert fallbacks_l == fallbacks_r

    # Verify chain-string alias is registered (semicolon-separated internal representation)
    chain_alias = "[claude-cli/claude-sonnet-5;github-copilot/claude-sonnet-5;openrouter/z-ai/glm-5.2]"
    chain_deployments = [d for d in model_list if d["model_name"] == chain_alias]
    assert [d["litellm_params"]["model"] for d in chain_deployments] == [
        "claude-cli/claude-sonnet-5",
        "github-copilot/gh-claude-sonnet-5",
        "openrouter/z-ai/glm-5.2",
    ]
    assert [d["litellm_params"]["order"] for d in chain_deployments] == [1, 2, 3]
    assert [d["model_info"]["rate_limit_key"] for d in chain_deployments] == [
        "claude-cli/claude-sonnet-5",
        "github-copilot/claude-sonnet-5",
        "openrouter/z-ai/glm-5.2",
    ]

    # Verify individual tokens are registered as deployments
    t1 = "claude-cli/claude-sonnet-5"
    assert any(d["model_name"] == t1 for d in model_list)
    t2 = "github-copilot/claude-sonnet-5"
    assert any(d["model_name"] == t2 for d in model_list)
    t3 = "openrouter/z-ai/glm-5.2"
    assert any(d["model_name"] == t3 for d in model_list)

    # Verify fallbacks are registered for the chain
    fallback_entry = next((f for f in fallbacks_l if chain_alias in f), None)
    assert fallback_entry is not None
    assert fallback_entry[chain_alias] == [t2, t3]

    # Verify small_model is registered as plain deployment/alias
    assert any(d["model_name"] == "github-copilot/gpt-5-mini" for d in model_list)
    
    # Verify router_settings
    assert router_settings["routing_strategy"] == "simple-shuffle"
    assert router_settings["num_retries"] == 0
    assert router_settings["disable_cooldowns"] is False


def test_parse_fallback_chain_with_account():
    # Single token with @account
    t1 = parse_fallback_chain("claude-cli@work/claude-opus-4-8")
    assert t1 == ["claude-cli@work/claude-opus-4-8"]

    # Bracketed chain with two @account tokens + one without
    t2 = parse_fallback_chain(
        "open-llm-proxy/[claude-cli@work/claude-opus-4-8,claude-cli@home/claude-opus-4-8,github-copilot/claude-opus-4.8]"
    )
    assert t2 == [
        "claude-cli@work/claude-opus-4-8",
        "claude-cli@home/claude-opus-4-8",
        "github-copilot/claude-opus-4.8",
    ]

    # Plain token without @ still works
    t3 = parse_fallback_chain("github-copilot/claude-sonnet-5")
    assert t3 == ["github-copilot/claude-sonnet-5"]


def test_parse_fallback_chain_account_validation():
    # Empty account after @
    with pytest.raises(ValueError, match="Malformed account tag"):
        parse_fallback_chain("claude-cli@/claude-opus-4-8")

    # Bad characters in account
    with pytest.raises(ValueError, match="Malformed account tag"):
        parse_fallback_chain("claude-cli@BAD!/claude-opus-4-8")

    # Account with uppercase (invalid per regex — must be lowercase)
    with pytest.raises(ValueError, match="Malformed account tag"):
        parse_fallback_chain("claude-cli@Work/claude-opus-4-8")


def test_map_token_to_deployment_params_with_account(monkeypatch):
    # claude-cli with @account emits claude_account
    params = map_token_to_deployment_params("claude-cli@work/claude-opus-4-8")
    assert params["model"] == "claude-cli/claude-opus-4-8"
    assert params["claude_account"] == "work"

    # claude-cli without @account emits NO claude_account
    params2 = map_token_to_deployment_params("claude-cli/claude-opus-4-8")
    assert params2["model"] == "claude-cli/claude-opus-4-8"
    assert "claude_account" not in params2

    # github-copilot with @account is not implemented → ValueError
    with pytest.raises(ValueError, match="not yet supported"):
        map_token_to_deployment_params("github-copilot@work/claude-sonnet-5")

    # nvidia_nim with @account but no stored secret → ValueError
    with pytest.raises(ValueError, match="no stored credential"):
        map_token_to_deployment_params("nvidia_nim@prod/nvidia/nemotron-4")

    # opencode with @account is not implemented → ValueError
    with pytest.raises(ValueError, match="not yet supported"):
        map_token_to_deployment_params("opencode@work/big-pickle")


def test_generate_config_with_account(tmp_path):
    dummy_yaml = """
file_settings:
  opencode:
    model: "open-llm-proxy/[claude-cli@work/claude-opus-4-8,claude-cli@home/claude-opus-4-8,github-copilot/claude-opus-4.8]"
    small_model: "claude-cli@default/claude-sonnet-5"
"""
    config_file = tmp_path / "agent-config.yml"
    config_file.write_text(dummy_yaml)
    config_dict = __import__("open_llm_proxy.config_gen", fromlist=[""]).generate_config(str(config_file))

    model_list = config_dict["model_list"]
    fallbacks_l = config_dict["litellm_settings"]["fallbacks"]
    fallbacks_r = config_dict["router_settings"]["fallbacks"]
    assert fallbacks_l == fallbacks_r

    # Chain alias uses ;-separated internal form, @ preserved
    chain_alias = (
        "[claude-cli@work/claude-opus-4-8;claude-cli@home/claude-opus-4-8;github-copilot/claude-opus-4.8]"
    )
    chain_deployments = [d for d in model_list if d["model_name"] == chain_alias]
    assert len(chain_deployments) == 3
    # Order 1: @work
    assert chain_deployments[0]["litellm_params"]["claude_account"] == "work"
    assert chain_deployments[0]["litellm_params"]["model"] == "claude-cli/claude-opus-4-8"
    assert chain_deployments[0]["model_info"]["rate_limit_key"] == "claude-cli@work/claude-opus-4-8"
    # Order 2: @home
    assert chain_deployments[1]["litellm_params"]["claude_account"] == "home"
    assert chain_deployments[1]["litellm_params"]["model"] == "claude-cli/claude-opus-4-8"
    assert chain_deployments[1]["model_info"]["rate_limit_key"] == "claude-cli@home/claude-opus-4-8"
    # Order 3: github-copilot (no account)
    assert "claude_account" not in chain_deployments[2]["litellm_params"]
    assert chain_deployments[2]["model_info"]["rate_limit_key"] == "github-copilot/claude-opus-4.8"

    # Individual token deployments registered with @ in key
    assert any(d["model_name"] == "claude-cli@work/claude-opus-4-8" for d in model_list)
    assert any(d["model_name"] == "claude-cli@home/claude-opus-4-8" for d in model_list)
    assert any(d["model_name"] == "github-copilot/claude-opus-4.8" for d in model_list)

    # small_model with @account
    assert any(d["model_name"] == "claude-cli@default/claude-sonnet-5" for d in model_list)

    # Fallbacks mapping: chain -> [tokens[1:]]
    fallback_entry = next((f for f in fallbacks_l if chain_alias in f), None)
    assert fallback_entry is not None
    assert fallback_entry[chain_alias] == [
        "claude-cli@home/claude-opus-4-8",
        "github-copilot/claude-opus-4.8",
    ]


def test_invalid_claude_cli_id():
    with pytest.raises(ValueError, match="Invalid claude-cli model ID"):
        map_token_to_deployment_params("claude-cli/invalid-model-name")


def test_supported_models_config_gen(tmp_path):
    dummy_yaml = """
file_settings:
  opencode:
    model: "github-copilot/gpt-5-mini"
    supported_models:
      - "claude-cli/claude-opus-4-8"
      - "claude-cli/claude-sonnet-5"
"""
    config_file = tmp_path / "agent-config.yml"
    config_file.write_text(dummy_yaml)
    config_dict = generate_config(str(config_file))
    model_list = config_dict["model_list"]

    # Verify that the supported models are registered standalone
    assert any(d["model_name"] == "claude-cli/claude-opus-4-8" for d in model_list)
    assert any(d["model_name"] == "claude-cli/claude-sonnet-5" for d in model_list)


# ---- HIGH — @account resolution for openrouter/nvidia -------------------------


class TestMapTokenWithStoredAccount:
    def test_openrouter_named_with_secret_injects_api_key(self, cfg, monkeypatch):
        """openrouter@work/model with a stored secret injects api_key."""
        from open_llm_proxy import account_registry

        account_registry.add_account(
            "openrouter", "work",
            storage="api-key", secret_bytes=b"sk-or-work-secret",
        )
        params = map_token_to_deployment_params("openrouter@work/z-ai/glm-5.2")
        assert params["model"] == "openrouter/z-ai/glm-5.2"
        assert params["api_key"] == "sk-or-work-secret"

    def test_openrouter_default_fallback_to_env(self, cfg, monkeypatch):
        """openrouter@default/model with env-line storage falls through to env file."""
        from open_llm_proxy import account_registry

        account_registry.add_account(
            "openrouter", "default", storage="env-line", ref="OPENROUTER_API_KEY",
        )
        monkeypatch.setattr(
            "open_llm_proxy.openrouter_creds.get_persisted_api_key",
            lambda account=None: "sk-or-from-env",
        )
        params = map_token_to_deployment_params("openrouter@default/z-ai/glm-5.2")
        assert params["model"] == "openrouter/z-ai/glm-5.2"
        assert params["api_key"] == "os.environ/OPENROUTER_API_KEY"

    def test_openrouter_named_no_secret_raises(self, cfg, monkeypatch):
        """openrouter@ghost/model with no stored secret raises ValueError."""
        with pytest.raises(ValueError, match="no stored credential"):
            map_token_to_deployment_params("openrouter@ghost/z-ai/glm-5.2")

    def test_nvidia_named_with_secret_injects_api_key(self, cfg, monkeypatch):
        """nvidia_nim@work/model with a stored secret injects api_key."""
        from open_llm_proxy import account_registry

        account_registry.add_account(
            "nvidia", "work",
            storage="api-key", secret_bytes=b"nv-work-secret",
        )
        params = map_token_to_deployment_params("nvidia_nim@work/nvidia/nemotron-4")
        assert params["model"] == "nvidia_nim/nvidia/nemotron-4"
        assert params["api_key"] == "nv-work-secret"


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set OLP_CONFIG_DIR to a tmp_path and return it."""
    d = tmp_path / "olp_config"
    d.mkdir()
    monkeypatch.setenv("OLP_CONFIG_DIR", str(d))
    return d
