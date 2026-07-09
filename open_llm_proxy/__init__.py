# open_llm_proxy package marker
from __future__ import annotations

import litellm
from litellm.utils import custom_llm_setup
from open_llm_proxy.provider_claude_cli import claude_cli_handler
from open_llm_proxy.provider_github_copilot import copilot_handler

if not hasattr(litellm, "custom_provider_map") or litellm.custom_provider_map is None:
    litellm.custom_provider_map = []

if not any(item.get("provider") == "claude-cli" for item in litellm.custom_provider_map):
    litellm.custom_provider_map.append({
        "provider": "claude-cli",
        "custom_handler": claude_cli_handler
    })

if not any(item.get("provider") == "github-copilot" for item in litellm.custom_provider_map):
    litellm.custom_provider_map.append({
        "provider": "github-copilot",
        "custom_handler": copilot_handler
    })

custom_llm_setup()

