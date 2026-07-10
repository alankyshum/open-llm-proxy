import argparse
import json
import os
import sys
from pathlib import Path
import yaml

def parse_fallback_chain(model_str: str) -> list[str]:
    if not isinstance(model_str, str):
        raise ValueError("model_str must be a string")
    s = model_str.strip()
    if s.startswith("open-llm-proxy/"):
        s = s[len("open-llm-proxy/"):]
    elif s.startswith("kilo-proxy/"):
        s = s[len("kilo-proxy/"):]
    
    if s.startswith("[") or s.endswith("]"):
        if not (s.startswith("[") and s.endswith("]")):
            raise ValueError(f"Malformed fallback chain: {model_str}")
        inner = s[1:-1].strip()
        if not inner:
            raise ValueError(f"Empty fallback chain brackets: {model_str}")
        tokens = [x.strip() for x in inner.split(",") if x.strip()]
        if not tokens:
            raise ValueError(f"Empty fallback chain brackets: {model_str}")
        for t in tokens:
            if "/" not in t:
                raise ValueError(f"Malformed fallback entry '{t}' in {model_str}: missing slash")
            parts = t.split("/")
            if any(not p.strip() for p in parts):
                raise ValueError(f"Malformed fallback entry '{t}' in {model_str}")
        return tokens
    else:
        if "/" not in s:
            raise ValueError(f"Malformed model string: {model_str}")
        parts = s.split("/")
        if any(not p.strip() for p in parts):
            raise ValueError(f"Malformed model string: {model_str}")
        return [s]

def map_token_to_deployment_params(token: str) -> dict:
    if "/" not in token:
        raise ValueError(f"Invalid token: {token}")
    provider, rest = token.split("/", 1)
    
    if provider == "claude-cli":
        return {
            "model": f"claude-cli/{rest}"
        }
    elif provider == "github-copilot":
        return {
            "model": f"github-copilot/gh-{rest}",
            "custom_llm_provider": "github-copilot",
            "api_key": "not-needed",
        }
    elif provider == "openrouter":
        return {
            "model": f"openrouter/{rest}",
            "api_key": "os.environ/OPENROUTER_API_KEY"
        }
    elif provider == "google":
        model_id = rest
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
        return {
            "model": f"gemini/{model_id}",
            "api_key": "os.environ/GEMINI_API_KEY"
        }
    elif provider == "opencode":
        return {
            "model": f"openai/{rest}",
            "api_base": "https://opencode.ai/zen/v1",
            "api_key": "os.environ/OPENCODE_API_KEY"
        }
    else:
        return {
            "model": token
        }

def generate_config(agent_config_path: str) -> dict:
    with open(agent_config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    
    model_strings = set()
    
    # Extract file_settings model strings
    file_settings = data.get("file_settings", {})
    if isinstance(file_settings, dict):
        opencode_settings = file_settings.get("opencode", {})
        if isinstance(opencode_settings, dict):
            if "model" in opencode_settings:
                model_strings.add(opencode_settings["model"])
            if "small_model" in opencode_settings:
                model_strings.add(opencode_settings["small_model"])
    
    # Extract agents model strings
    agents = data.get("agents", {})
    if isinstance(agents, dict):
        for agent_name, agent_cfg in agents.items():
            if isinstance(agent_cfg, dict) and "model" in agent_cfg:
                model_strings.add(agent_cfg["model"])
                
    deployments = {}
    fallbacks = []
    
    for raw_model in sorted(model_strings):
        tokens = parse_fallback_chain(raw_model)
        
        # Determine the key/alias for this chain/model
        stripped_model = raw_model
        if stripped_model.startswith("open-llm-proxy/"):
            stripped_model = stripped_model[len("open-llm-proxy/"):]
        elif stripped_model.startswith("kilo-proxy/"):
            stripped_model = stripped_model[len("kilo-proxy/"):]
            
        primary_token = tokens[0]
        
        # If it's a bracketed chain (length of tokens > 1 or raw_model is bracketed)
        is_bracketed = stripped_model.startswith("[") and stripped_model.endswith("]")
        if is_bracketed:
            internal_alias = stripped_model.replace(",", ";")
            deployments[internal_alias] = {
                "model_name": internal_alias,
                "litellm_params": map_token_to_deployment_params(primary_token)
            }
            if len(tokens) > 1:
                # Add fallbacks mapping
                fallbacks.append({
                    internal_alias: tokens[1:]
                })
        else:
            # Plain model, just register the plain model deployment
            deployments[stripped_model] = {
                "model_name": stripped_model,
                "litellm_params": map_token_to_deployment_params(primary_token)
            }
            
        # Register every token as its own deployment
        for token in tokens:
            if token not in deployments:
                deployments[token] = {
                    "model_name": token,
                    "litellm_params": map_token_to_deployment_params(token)
                }
                
    # Sort deployments by model_name for deterministic output
    model_list = [deployments[k] for k in sorted(deployments.keys())]
    
    router_settings = {
        "num_retries": 3,
        "cooldown_time": 30,
        "routing_strategy": "simple-shuffle",
        "fallbacks": fallbacks
    }

    litellm_settings = {
        "fallbacks": fallbacks
    }
    
    return {
        "model_list": model_list,
        "litellm_settings": litellm_settings,
        "router_settings": router_settings
    }

def main():
    parser = argparse.ArgumentParser(description="Generate LiteLLM Router config from agent-config.yml")
    parser.add_argument("--config-path", default="../../../config/agent-runtime/agent-config.yml", help="Path to agent-config.yml")
    parser.add_argument("--print", choices=["json", "yaml"], default="yaml", help="Print format")
    args = parser.parse_args()
    
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        resolved_path = Path(os.getcwd()) / config_path
        if not resolved_path.exists():
            resolved_path = Path(__file__).resolve().parent / config_path
        config_path = resolved_path
        
    try:
        config_dict = generate_config(str(config_path))
        if args.print == "json":
            print(json.dumps(config_dict, indent=2))
        else:
            print(yaml.safe_dump(config_dict, sort_keys=False))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
