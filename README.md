# open-llm-proxy

`open-llm-proxy` is a local OpenAI-compatible gateway for routing requests through the models and provider accounts you already use. It gives agents one stable endpoint, configurable fallback chains, provider authentication, attachment handling, rate-limit tracking, and an optional Admin UI backed by PostgreSQL.

## Features

- OpenAI-compatible `/v1/chat/completions` and `/healthz` endpoints.
- Provider/model fallback chains and model discovery.
- Named provider accounts with safe credential storage.
- Attachment normalization and optional on-disk spooling.
- Persistent provider rate-limit policy and state.
- Optional LiteLLM Admin UI for keys, requests, and spend tracking.

## Requirements

- Python 3.10 or newer.
- `uv` or `pip`.
- Provider credentials for the models in your configuration.
- PostgreSQL only if you want the Admin UI.

## Install

```bash
uv tool install .
# or
python -m pip install .
```

## Quickstart (5 minutes)

```bash
mkdir -p ~/.config/open-llm-proxy
cat > ~/.config/open-llm-proxy/agent-config.yml <<'YAML'
opencode:
  settings:
    model: "openrouter/<provider-model-id>"
YAML
export OPENROUTER_API_KEY="<your-provider-key>"
open-llm-proxy serve
```

In another terminal, check readiness and send a request:

```bash
curl http://127.0.0.1:8765/healthz
curl http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openrouter/<provider-model-id>","messages":[{"role":"user","content":"Hello"}]}'
```

The default bind is `127.0.0.1`. Remote binding is an explicit opt-in and is unauthenticated unless `LITELLM_MASTER_KEY` is set. Pass secrets through environment variables; CLI secret options remain available but are deprecated.

Attachments are stored under `~/.local/share/open-llm-proxy/attachments` by default. Provider account metadata and credentials are stored under `~/.config/open-llm-proxy/` with restricted permissions. Read [SECURITY.md](SECURITY.md) before exposing the service.

## Admin UI with PostgreSQL

For the optional Admin UI, copy `.env.example` to `.env`, fill in the PostgreSQL and UI credentials, and load that environment before starting the proxy. Keep `.env` private.

## Documentation

| Guide | Contents |
| --- | --- |
| [Getting started](docs/getting-started.md) | Full setup, Admin UI, PostgreSQL, and Podman |
| [CLI](docs/cli.md) | Commands and options |
| [Authentication](docs/authentication.md) | Accounts, credentials, and migration |
| [Attachments](docs/attachments.md) | Normalization and file spooling |
| [Rate limits](docs/rate-limits.md) | Plans, setup, and cooldowns |
| [Configuration](docs/configuration.md) | Config files and environment variables |

## License and contributing

MIT; see [LICENSE](LICENSE). Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
