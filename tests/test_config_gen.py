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

def test_map_token_to_deployment_params():
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


def test_invalid_claude_cli_id():
    with pytest.raises(ValueError, match="Invalid claude-cli model ID"):
        map_token_to_deployment_params("claude-cli/invalid-model-name")


def test_surfaced_models_config_gen(tmp_path):
    dummy_yaml = """
file_settings:
  opencode:
    model: "github-copilot/gpt-5-mini"
    surfaced_models:
      - "claude-cli/claude-opus-4-8"
      - "claude-cli/claude-sonnet-5"
"""
    config_file = tmp_path / "agent-config.yml"
    config_file.write_text(dummy_yaml)
    config_dict = generate_config(str(config_file))
    model_list = config_dict["model_list"]

    # Verify that the surfaced models are registered standalone
    assert any(d["model_name"] == "claude-cli/claude-opus-4-8" for d in model_list)
    assert any(d["model_name"] == "claude-cli/claude-sonnet-5" for d in model_list)
