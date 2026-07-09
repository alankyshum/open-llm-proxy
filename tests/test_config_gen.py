import pytest
from pathlib import Path
from open_llm_proxy.config_gen import (
    parse_fallback_chain,
    map_token_to_deployment_params,
    generate_config,
)

def test_parse_fallback_chain_success():
    # dual prefix acceptance
    c1 = "kilo-proxy/[claude-cli/claude-sonnet-5,github-copilot/claude-sonnet-5,openrouter/z-ai/glm-5.2]"
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
    assert params_copilot["model"] == "github-copilot/claude-sonnet-5"
    assert params_copilot["api_key"] == "sk-copilot-local"

def test_generate_config_real(tmp_path):
    # Create a dummy agent-config.yml
    dummy_yaml = """
file_settings:
  opencode:
    model: "kilo-proxy/[claude-cli/claude-sonnet-5,github-copilot/claude-sonnet-5,openrouter/z-ai/glm-5.2]"
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
    assert "fallbacks" in config_dict
    assert "router_settings" in config_dict

    model_list = config_dict["model_list"]
    fallbacks = config_dict["fallbacks"]
    router_settings = config_dict["router_settings"]

    # Verify chain-string alias is registered (semicolon-separated internal representation)
    chain_alias = "[claude-cli/claude-sonnet-5;github-copilot/claude-sonnet-5;openrouter/z-ai/glm-5.2]"
    chain_deployment = next((d for d in model_list if d["model_name"] == chain_alias), None)
    assert chain_deployment is not None
    # Points to first token: claude-cli/claude-sonnet-5
    assert chain_deployment["litellm_params"]["model"] == "claude-cli/claude-sonnet-5"

    # Verify individual tokens are registered as deployments
    t1 = "claude-cli/claude-sonnet-5"
    assert any(d["model_name"] == t1 for d in model_list)
    t2 = "github-copilot/claude-sonnet-5"
    assert any(d["model_name"] == t2 for d in model_list)
    t3 = "openrouter/z-ai/glm-5.2"
    assert any(d["model_name"] == t3 for d in model_list)

    # Verify fallbacks are registered for the chain
    fallback_entry = next((f for f in fallbacks if chain_alias in f), None)
    assert fallback_entry is not None
    assert fallback_entry[chain_alias] == [t2, t3]

    # Verify small_model is registered as plain deployment/alias
    assert any(d["model_name"] == "github-copilot/gpt-5-mini" for d in model_list)
    
    # Verify router_settings
    assert router_settings["routing_strategy"] == "simple-shuffle"
    assert router_settings["num_retries"] == 3
