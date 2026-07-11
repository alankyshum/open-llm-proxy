# open-llm-proxy
Lightweight litellm Router proxy

## CLI

Installing the package exposes the `open-llm-proxy` command:

```bash
open-llm-proxy serve --host 127.0.0.1 --port 8765
open-llm-proxy setup --config ~/.config/open-llm-proxy/agent-config.yml
open-llm-proxy config --format yaml
open-llm-proxy models openrouter --search kimi
open-llm-proxy models opencode --format json
open-llm-proxy ui --open
open-llm-proxy help serve
```

Run `open-llm-proxy help` to dynamically list available commands, or
`open-llm-proxy help <command>` to show that command's current options.
`--help` remains available at every level. The legacy
`python -m open_llm_proxy.*` entry points remain available.

`models` shells out to OpenCode's provider catalog, so its output uses exact IDs
accepted by `agent-config.yml`. Omit the provider to list every catalog, filter
with case-insensitive `--search`, or use `--format json` from scripts. The
`available-models` alias is equivalent.

## LiteLLM Admin UI (first-class, on by default)

`open-llm-proxy` ships the LiteLLM Admin UI as a first-class feature for virtual
key management and request/spend tracking. **The UI is enabled by default** — it
only needs a PostgreSQL database. The proxy is fail-safe: if no `DATABASE_URL` is
configured it automatically degrades to a stateless DB-less router (UI off)
instead of crashing, and if a database is present but no master key is set it
auto-provisions an ephemeral one.

### 1. Start PostgreSQL (Podman)

A `docker-compose.yml` in this repo provisions the database. Manage it with
Podman:

```bash
podman compose up -d      # start Postgres (litellm/litellm @ 127.0.0.1:5432)
podman compose down       # stop it
```

### 2. Point the proxy at the database

Set these in `~/.config/open-llm-proxy/env` (the deploy consumes this file):

```bash
DATABASE_URL=postgresql://litellm:litellm@127.0.0.1:5432/litellm
LITELLM_MASTER_KEY=sk-...        # optional; auto-generated if omitted
UI_USERNAME=admin                # Admin UI login
UI_PASSWORD=admin
STORE_MODEL_IN_DB=True
```

On startup the launcher generates the Prisma client and pushes the LiteLLM
schema to the database automatically (self-healing across redeploys). The Admin
UI is then served at `http://127.0.0.1:8765/ui`.

Verify / open it:
```bash
open-llm-proxy ui --open       # checks /ui and opens your browser
```

Or run ad-hoc with explicit flags (flags win over env; existing secrets are
never overwritten):
```bash
open-llm-proxy serve \
  --host 127.0.0.1 --port 8765 \
  --master-key "sk-my-secure-master-key" \
  --database-url "postgresql://litellm:litellm@127.0.0.1:5432/litellm" \
  --ui-username "admin" --ui-password "secure-password"
```

### 3. Stateless / DB-less mode

To run a lightweight router with no database (UI off), either provide no
`DATABASE_URL` (auto-degrades) or disable it explicitly:

```bash
open-llm-proxy serve --disable-admin-ui
# or
export DISABLE_ADMIN_UI="True" && open-llm-proxy serve
```

## Provider rate-limit setup

Run setup after installing the proxy:

```bash
python -m open_llm_proxy.setup \
  --config ~/.config/open-llm-proxy/agent-config.yml
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
