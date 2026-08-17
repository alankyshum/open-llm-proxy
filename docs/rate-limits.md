# Rate limits

Run setup after configuring models:

```bash
open-llm-proxy setup --config <path>
```

Setup discovers provider/model pairs, records the selected provider plan and its source policy, and stores observed cooldown state in SQLite. Use `--force` to replace an existing plan or `--non-interactive` to use configured/default plans.

Provider `Retry-After` and reset headers take precedence. If an upstream response has no reset metadata, the selected plan's fallback cooldown is used. Account-qualified tokens such as `claude-cli@work/model` keep separate cooldown buckets while sharing the provider plan.

The built-in catalog covers Claude subscriptions and API tiers, GitHub Copilot, Gemini, OpenRouter, OpenCode Zen, and local Ollama. Catalog entries include a source URL and verification date.
