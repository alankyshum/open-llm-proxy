# Authentication and accounts

Manage provider credentials without putting secrets in shell history:

```bash
open-llm-proxy auth
open-llm-proxy auth --no-tui
open-llm-proxy auth add openrouter
open-llm-proxy auth add nvidia --name production
open-llm-proxy auth accounts
open-llm-proxy auth use claude-cli work
open-llm-proxy auth rename openrouter default work
open-llm-proxy auth remove nvidia production
open-llm-proxy auth check
```

Credentials are collected through hidden prompts, stdin, or provider OAuth helpers. Do not pass provider secrets as CLI arguments. The first account is named `default`; additional accounts use `--name`. Named accounts can be selected in a chain as `provider@account/model`. Without `@account`, the active account is used. Accounts have independent rate-limit buckets and fallback slots.

Supported providers include OpenRouter, NVIDIA, OpenCode, GitHub Copilot, and Claude CLI. The legacy `auth set <provider>` command remains available for single-account setup.

## Storage and migration

- Registry: `~/.config/open-llm-proxy/accounts.json` (metadata only, mode `0600`).
- Per-account secrets: `~/.config/open-llm-proxy/accounts/<provider>/` (mode `0600`).
- Override the base directory with `OLP_CONFIG_DIR`.

On the first authentication operation, existing discoverable credentials are imported as `default` accounts. Migration is one-time, idempotent, and non-destructive. API-key providers loaded from the shared environment file require a proxy restart after changes; OAuth credentials are read at runtime.
