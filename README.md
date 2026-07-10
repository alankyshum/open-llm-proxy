# open-llm-proxy
Lightweight litellm Router proxy

## CLI

Installing the package exposes the `open-llm-proxy` command:

```bash
open-llm-proxy serve --host 127.0.0.1 --port 8765
open-llm-proxy setup --config ~/.config/kilo-claude-proxy/agent-config.yml
open-llm-proxy config --format yaml
open-llm-proxy help serve
```

Run `open-llm-proxy help` to dynamically list available commands, or
`open-llm-proxy help <command>` to show that command's current options.
`--help` remains available at every level. The legacy
`python -m open_llm_proxy.*` entry points remain available.

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
