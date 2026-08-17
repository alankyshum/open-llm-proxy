# Configuration

## Agent configuration

The service reads YAML from `agent-config.yml`. Put model selectors under `opencode.settings.model`, `opencode.settings.small_model`, `opencode.settings.supported_models`, or `opencode.agents.<name>.model`. A fallback chain can use the form `open-llm-proxy/[provider/model-a,provider/model-b]`.

Configuration resolution, highest priority first:

1. `--config <path>` (also accepted as `--config-path`).
2. `OPEN_LLM_PROXY_CONFIG` (must point to an existing file; it does not fall through if missing).
3. `$XDG_CONFIG_HOME/open-llm-proxy/agent-config.yml` (default: `~/.config/open-llm-proxy/agent-config.yml`).
4. `./agent-config.yml`.

## Environment variables

Set secrets in the environment or in a private environment file. CLI secret flags are deprecated.

| Variable | Purpose |
| --- | --- |
| `OPEN_LLM_PROXY_CONFIG` | Agent configuration path. |
| `OPEN_LLM_PROXY_HOST` | Bind address for the proxy server (default: `127.0.0.1`). |
| `XDG_CONFIG_HOME` | Base directory for the portable default configuration path. |
| `DATABASE_URL` | PostgreSQL connection string; enables the Admin UI. |
| `LITELLM_MASTER_KEY` | Master key for proxy/UI authentication. |
| `UI_USERNAME` | Admin UI username. |
| `UI_PASSWORD` | Admin UI password. |
| `DISABLE_ADMIN_UI` | Set to `True` or `1` to disable the Admin UI. |
| `OPEN_LLM_PROXY_CONFIG_DIR` | Canonical override for the proxy configuration, account registry, credential storage, runtime state, and generated config directory. |
| `OLP_CONFIG_DIR` | Backward-compatible alias for `OPEN_LLM_PROXY_CONFIG_DIR`. |
| `BYPASS_KEYCHAIN` | Set to `1` to skip macOS Keychain storage. Credentials fall back to plaintext-on-disk storage, reducing protection for stored secrets. |
| `COPILOT_OAUTH_TOKEN` | GitHub Copilot OAuth token. |
| `OPENCODE_API_KEY` | OpenCode API key. |
| `OPENCODE_AUTH_PATH` | Path to the OpenCode authentication file. |
| `NVIDIA_API_KEY` | NVIDIA provider credential; legacy environment import is supported. |
| `OPEN_LLM_PROXY_BILLING_HEADER` | Override the billing header used for requests. |
| `OPEN_LLM_PROXY_RTK_GIT` | Set to `1` to enable RTK Git integration. |
| `OPEN_LLM_PROXY_GOOGLE_MIN_MAX_TOKENS` | Enable the configured Gemini/Google minimum max-token behavior. |
| `OPEN_LLM_PROXY_STICKY_ROUTING` | Set to `1` to enable sticky routing. |
| `OPEN_LLM_PROXY_ATTRIBUTION_TOKEN` | Attribution token. |
| `OPEN_LLM_PROXY_ATTRIBUTION_TOKEN_FILE` | File containing the attribution token. |

Attachment-specific variables are documented in [Attachments](attachments.md).
