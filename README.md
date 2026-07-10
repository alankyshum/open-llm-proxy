# open-llm-proxy
Lightweight litellm Router proxy

## Provider rate-limit setup

Run setup after installing the proxy:

```bash
python -m open_llm_proxy.setup \
  --config ~/.config/kilo-claude-proxy/agent-config.yml
```

Setup discovers every provider/model pair in `agent-config.yml`, asks which
provider plan applies, and stores the selected plan, sourced policy, model
inventory, and observed rate-limit state in SQLite. Reconfigure existing plans
with `--force`.

At runtime, an HTTP `Retry-After` or rate-limit reset header takes precedence.
When the provider does not return reset metadata, the selected plan's sourced
fallback cooldown is used. The built-in catalog covers Claude subscriptions,
GitHub Copilot, Gemini API tiers, OpenRouter, and OpenCode Zen. Catalog entries
include their source URL and verification date; Gemini's model/project-specific
limits link to AI Studio because Google does not publish one fixed static quota.
