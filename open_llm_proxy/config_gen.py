import argparse
import json
import os
import re
import sys
from pathlib import Path
import yaml

# Import the catalog from translator to validate supported models
try:
    from open_llm_proxy.translator import _MODEL_CATALOG
    _CATALOG_IDS = {m["id"] for m in _MODEL_CATALOG}
except ImportError:
    _CATALOG_IDS = {
        "claude-opus-4-8", "claude-sonnet-5", "claude-fable-5",
        "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"
    }

_ACCOUNT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _validate_account_in_token(token: str, context: str) -> None:
    """Validate @account suffix in the provider portion. Raise ValueError if malformed."""
    provider_part = token.split("/")[0]
    if "@" in provider_part:
        _discard, account = provider_part.split("@", 1)
        if not account or not _ACCOUNT_RE.match(account):
            raise ValueError(
                f"Malformed account tag in '{token}' in {context}"
            )


def parse_fallback_chain(model_str: str) -> list[str]:
    if not isinstance(model_str, str):
        raise ValueError("model_str must be a string")
    s = model_str.strip()
    if s.startswith("open-llm-proxy/"):
        s = s[len("open-llm-proxy/"):]
    
    if s.startswith("[") or s.endswith("]") :
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
            _validate_account_in_token(t, model_str)
        return tokens
    else:
        if "/" not in s:
            raise ValueError(f"Malformed model string: {model_str}")
        parts = s.split("/")
        if any(not p.strip() for p in parts):
            raise ValueError(f"Malformed model string: {model_str}")
        _validate_account_in_token(s, model_str)
        return [s]

def _resolve_account_api_key(
    provider_registry_key: str, account: str, token: str,
) -> str:
    """Resolve a named account's API key from registry per-account file.

    Raises ``ValueError`` (with a reference to *token*) if the account
    has no stored credential.
    """
    from open_llm_proxy import account_registry

    try:
        secret = account_registry.read_secret(provider_registry_key, account)
    except account_registry.AccountRegistryError:
        raise ValueError(
            f"account {account!r} for provider {provider_registry_key} "
            f"(in '{token}') has no stored credential"
        )
    if secret is None:
        raise ValueError(
            f"account {account!r} for provider {provider_registry_key} "
            f"(in '{token}') has no stored credential"
        )
    return secret.decode().strip()


def map_token_to_deployment_params(token: str) -> dict:
    if "/" not in token:
        raise ValueError(f"Invalid token: {token}")
    provider_and_account, rest = token.split("/", 1)
    
    # Split optional @account from provider
    account = None
    if "@" in provider_and_account:
        base_provider, account = provider_and_account.split("@", 1)
    else:
        base_provider = provider_and_account
    
    if base_provider == "claude-cli":
        base_model_id = rest.split(":")[0] if ":" in rest else rest
        if base_model_id not in _CATALOG_IDS:
            raise ValueError(f"Invalid claude-cli model ID: {base_model_id}")
        result: dict = {
            "model": f"claude-cli/{rest}"
        }
        if account is not None:
            result["claude_account"] = account
        return result
    elif base_provider == "github-copilot":
        if account is not None:
            raise ValueError(
                f"account {account!r} for provider github-copilot "
                f"(in '{token}') is not yet supported. "
                f"Use github-copilot/{rest} (without @account) for the default account."
            )
        return {
            "model": f"github-copilot/gh-{rest}",
            "custom_llm_provider": "github-copilot",
            "api_key": "not-needed",
        }
    elif base_provider == "openrouter":
        if account is not None and account != "default":
            key = _resolve_account_api_key("openrouter", account, token)
            return {"model": f"openrouter/{rest}", "api_key": key}
        from open_llm_proxy.openrouter_creds import get_persisted_api_key
        key = get_persisted_api_key()
        os.environ["OPENROUTER_API_KEY"] = key
        return {
            "model": f"openrouter/{rest}",
            "api_key": "os.environ/OPENROUTER_API_KEY"
        }
    elif base_provider in ("nvidia", "nvidia_nim"):
        if account is not None and account != "default":
            key = _resolve_account_api_key("nvidia", account, token)
            return {"model": f"nvidia_nim/{rest}", "api_key": key}
        return {
            "model": f"nvidia_nim/{rest}",
        }
    elif base_provider == "google":
        model_id = rest
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
        return {
            "model": f"gemini/{model_id}",
            "api_key": "os.environ/GEMINI_API_KEY"
        }
    elif base_provider == "opencode":
        if account is not None:
            raise ValueError(
                f"account {account!r} for provider opencode "
                f"(in '{token}') is not yet supported. "
                f"Use opencode/{rest} (without @account) for the default account."
            )
        from open_llm_proxy.opencode_creds import get_opencode_api_key
        # Securely read key at runtime/config generation to fail fast if absent.
        # This populates os.environ["OPENCODE_API_KEY"] if sourced from auth.json.
        key = get_opencode_api_key()
        os.environ["OPENCODE_API_KEY"] = key
        return {
            "model": f"openai/{rest}",
            "api_base": "https://opencode.ai/zen/v1",
            "api_key": "os.environ/OPENCODE_API_KEY"
        }
    elif base_provider == "ollama-local":
        # Local Ollama server — no auth, no remote quota. litellm dispatches
        # via its ollama_chat provider against the local daemon. num_ctx caps
        # the KV-cache: the model's max context (262k) balloons RAM to ~26GB,
        # so we pin a modest window suitable for vision/UI tasks. think=False
        # disables the reasoning/thinking block for controllable-reasoning
        # models (e.g. gemma4) so the final answer isn't preceded by a think
        # block the opencode client drops; num_predict is a generous cap.
        return {
            "model": f"ollama_chat/{rest}",
            "api_base": "http://127.0.0.1:11434",
            "num_ctx": 16384,
            "num_predict": 4000,
            "think": False,
        }
    else:
        return {
            "model": f"{base_provider}/{rest}"
        }


def parse_agent_config(source: str | bytes) -> dict:
    """Parse one immutable agent-config snapshot."""
    data = yaml.safe_load(source) or {}
    if not isinstance(data, dict):
        raise ValueError("agent config must be a mapping")
    return data


def configured_model_tokens_from_data(data: dict) -> set[str]:
    """Extract concrete provider/model keys from parsed agent config."""

    model_strings = set()
    
    # Extract opencode.settings model strings
    opencode_settings = (data.get("opencode") or {}).get("settings", {})
    if isinstance(opencode_settings, dict):
        if "model" in opencode_settings:
            model_strings.add(opencode_settings["model"])
        if "small_model" in opencode_settings:
            model_strings.add(opencode_settings["small_model"])
        # Extract supported_models
        supported_models = opencode_settings.get("supported_models", [])
        if isinstance(supported_models, list):
            for m in supported_models:
                if isinstance(m, str):
                    # Supported models are bare tokens like "claude-cli/claude-sonnet-5"
                    # We need to wrap them as if they were plain models
                    model_strings.add(f"open-llm-proxy/{m}")
    
    # Extract agents model strings
    agents = data.get("agents", {})
    if isinstance(agents, dict):
        for agent_name, agent_cfg in agents.items():
            if isinstance(agent_cfg, dict) and "model" in agent_cfg:
                model_strings.add(agent_cfg["model"])

    tokens: set[str] = set()
    for raw_model in model_strings:
        tokens.update(parse_fallback_chain(raw_model))
    return tokens


def configured_model_tokens(agent_config_path: str | Path) -> set[str]:
    return configured_model_tokens_from_data(
        parse_agent_config(Path(agent_config_path).read_bytes())
    )


def generate_config_from_data(data: dict) -> dict:
    """Generate LiteLLM config from one parsed agent-config snapshot."""

    model_strings = set()
    opencode_settings = (data.get("opencode") or {}).get("settings", {})
    if isinstance(opencode_settings, dict):
        for key in ("model", "small_model"):
            if key in opencode_settings:
                model_strings.add(opencode_settings[key])
        for model in opencode_settings.get("supported_models", []):
            if isinstance(model, str):
                model_strings.add(f"open-llm-proxy/{model}")
    for agent_cfg in (data.get("agents") or {}).values():
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
            
        primary_token = tokens[0]
        
        # If it's a bracketed chain (length of tokens > 1 or raw_model is bracketed)
        is_bracketed = stripped_model.startswith("[") and stripped_model.endswith("]")
        if is_bracketed:
            internal_alias = stripped_model.replace(",", ";")
            for order, token in enumerate(tokens, start=1):
                params = map_token_to_deployment_params(token)
                params["order"] = order
                deployments[f"{internal_alias}::{order}"] = {
                    "model_name": internal_alias,
                    "litellm_params": params,
                    "model_info": {"rate_limit_key": token},
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
                "litellm_params": map_token_to_deployment_params(primary_token),
                "model_info": {"rate_limit_key": primary_token},
            }
            
        # Register every token as its own deployment
        for token in tokens:
            if token not in deployments:
                deployments[token] = {
                    "model_name": token,
                    "litellm_params": map_token_to_deployment_params(token),
                    "model_info": {"rate_limit_key": token},
                }
                
        # Register supported models that are bare tokens (not in chains)
        # These come from supported_models list as bare tokens like "claude-cli/claude-sonnet-5"
        # They don't have open-llm-proxy/ prefix because they come from supported_models list directly
        # But since we added "open-llm-proxy/{m}" to model_strings, they've already been processed above

    # Sort deployments by model_name for deterministic output
    model_list = [deployments[k] for k in sorted(deployments.keys())]

    router_settings = {
        "num_retries": 0,
        "disable_cooldowns": False,
        "routing_strategy": "simple-shuffle",
        "fallbacks": fallbacks
    }

    litellm_settings = {
        "fallbacks": fallbacks,
        "drop_params": True,
    }
    
    return {
        "model_list": model_list,
        "litellm_settings": litellm_settings,
        "router_settings": router_settings
    }


def generate_config(agent_config_path: str) -> dict:
    return generate_config_from_data(
        parse_agent_config(Path(agent_config_path).read_bytes())
    )

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
