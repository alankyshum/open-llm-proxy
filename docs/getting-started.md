# Getting started

## Install and run locally

Install with `uv tool install .` or `python -m pip install .`. Create an `agent-config.yml` containing at least one model under `opencode.settings.model` or `opencode.agents.<name>.model`, then authenticate that provider with `open-llm-proxy auth add <provider>`.

The portable configuration lookup order is documented in [Configuration](configuration.md). Start the service with:

```bash
open-llm-proxy serve
```

The default listener is `127.0.0.1:8765`. Verify it with `curl http://127.0.0.1:8765/healthz`. Requests use `/v1/chat/completions` and the model IDs from your configuration.

## Admin UI with PostgreSQL

The Admin UI is available when a PostgreSQL connection is configured. Copy `.env.example` to `.env`, fill in the PostgreSQL values and UI credentials with your own values, and load that environment before starting the proxy. Keep `.env` private.

With Docker Compose:

```bash
docker compose up -d
open-llm-proxy serve
```

With Podman:

```bash
podman compose up -d
open-llm-proxy serve
```

The database port is bound to localhost. Open `http://127.0.0.1:8765/ui`, or run `open-llm-proxy ui --open`. Without a database, the proxy runs DB-less and the UI is disabled. The UI can also be disabled with `open-llm-proxy serve --disable-admin-ui`.

Use `open-llm-proxy serve --host <address>` only when remote access is intentional. Remote binding has no authentication unless a master key is configured.

## Provider setup and discovery

`open-llm-proxy auth` opens the interactive credential flow when attached to a terminal. Use `open-llm-proxy auth --no-tui` for the fallback prompt. `open-llm-proxy models --search <text>` lists provider catalog IDs; copy an exact ID into `agent-config.yml`.

For rate-limit policy setup, run `open-llm-proxy setup --config <path>`. See [Rate limits](rate-limits.md).
