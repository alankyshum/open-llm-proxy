# open-llm-proxy
open-llm-proxy is a LiteLLM-based Router proxy with an open-llm-proxy CLI for serving, setup, configuration, model discovery, and the Admin UI. It ships the LiteLLM Admin UI by default for virtual-key management and request/spend tracking, while safely degrading to a stateless DB-less router when DATABASE_URL is not configured.

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

## Attachment content normalization

The proxy normalizes non-standard chat attachment parts before forwarding them
upstream, because providers accept only the OpenAI `text` and `image_url` part
types. Standard `text` and `image_url` parts are retained; image attachments
become `image_url` parts, and text-like data URIs are decoded inline. This is
shape- and MIME-driven rather than tied to specific models or providers.
Images sent to `/responses`-routed GitHub Copilot models are translated to the
`input_image` shape.

Normalization is enabled by default. Set
`OPEN_LLM_PROXY_NORMALIZE_ATTACHMENTS=0` (also `false` or `no`) to disable it.

### Path spooling for non-renderable attachments

Anything that is neither text-like nor an image — a PDF, an archive, a binary
blob — cannot be inlined. Instead of emitting a dead placeholder, the proxy
**writes the decoded bytes to disk** and hands the agent the absolute path:

```
[attachment: invoice.pdf (application/pdf), 48213 bytes]
Saved to: /Users/you/.local/share/open-llm-proxy/attachments/ab12cd34ef567890-invoice.pdf
Read this file from disk to access its contents.
```

The agent then uses its own file-reading tools on that path. The proxy performs
no format-specific parsing and makes no assumptions about model capabilities.

Spooled filenames are **content-addressed** — `<sha256[:16]>-<safe-name>` — so a
fallback chain retrying the same request against the next model reuses the
existing file rather than writing duplicates. Writes are atomic, the spool
directory is created `0700` and files are written `0600`. Spooling is entirely
best effort: on any failure the old descriptive placeholder is used instead.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `OPEN_LLM_PROXY_SPOOL_ATTACHMENTS` | `1` | Kill switch. Set to `0`/`false`/`no` to restore the plain placeholder. |
| `OPEN_LLM_PROXY_ATTACHMENT_DIR` | `~/.local/share/open-llm-proxy/attachments` | Where spooled attachments are written. |
| `OPEN_LLM_PROXY_ATTACHMENT_RETENTION_DAYS` | `7` | Spooled files older than this are pruned on each write. `0` disables pruning. |

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

## Authentication & accounts

Manage credentials for providers the router uses. The system supports
**multiple named accounts per provider** with independent rate-limit buckets.

```bash
open-llm-proxy auth                          # interactive TUI (TTY)
open-llm-proxy auth --no-tui                 # fallback numbered prompt
open-llm-proxy auth add openrouter           # add first (default) account
open-llm-proxy auth add nvidia --name prod   # add a named account
open-llm-proxy auth accounts                 # list all accounts
open-llm-proxy auth accounts claude-cli      # list one provider's accounts
open-llm-proxy auth use claude-cli work      # switch active account
open-llm-proxy auth rename openrouter default work   # rename (needs ≥2)
open-llm-proxy auth remove nvidia prod       # remove an account
open-llm-proxy auth remove nvidia default --force    # force-remove last
open-llm-proxy auth check                    # live probe all providers
open-llm-proxy auth check openrouter         # probe one provider
```

Secrets are always acquired via a hidden prompt, piped stdin, or an external
OAuth helper — **never** passed as command-line arguments.

### Interactive TUI

Running `open-llm-proxy auth` with a TTY opens an interactive menu
(powered by `questionary`, a base dependency). Pick a provider from the
list, paste an API key (hidden input) or launch an OAuth login helper.
Account naming is only prompted when a second account already exists.

Use `--no-tui` to skip the TUI and get the original numbered-prompt flow,
or pipe stdin for scripting.

### Multi-account model

Each provider credential lives under a **named account**:

- **First account** per provider is auto-named `default`.
- **Add more** with `auth add <provider> --name <name>`.
  `--name` is required when the provider already has accounts.
- **List** with `auth accounts [provider]`. The active account is marked
  with `*`.
- **Switch active** with `auth use <provider> <name>`.
- **Rename** with `auth rename <provider> <old> <new>` — only allowed once
  a second account exists for that provider.
- **Remove** with `auth remove <provider> <name>`. Removing the last
  account requires `--force`.

> The legacy `auth set <provider>` still works for quick single-account
> setup but does not support named accounts. Prefer `auth add <provider>`
> or the interactive TUI.

### Token syntax: `provider@account/model`

In agent-config chains, include the account with an `@` suffix on the
provider:

```
open-llm-proxy/[claude-cli@work/claude-opus-4-8,claude-cli@home/claude-opus-4-8,github-copilot/claude-opus-4.8]
```

- **Absent `@account`** → the provider's active account (or `default`)
  is used.
- **Each account is a separate deployment** with its own rate-limit bucket
  and failover slot.

### Supported providers

| Provider | Auth kind | Command example |
|---|---|---|
| OpenRouter | api-key | `auth add openrouter` |
| NVIDIA (NIM) | api-key | `auth add nvidia` |
| OpenCode | oauth-cli | `auth add opencode` |
| GitHub Copilot | oauth-cli | `auth add github-copilot` |
| Claude CLI | oauth-cli | `auth add claude-cli` |

`nvidia` is now a first-class provider — `auth add nvidia` persists your
key via the shared env-file writer. The older `NVIDIA_API_KEY` environment
variable still works and is automatically imported as the `default` account
on first auth invocation.

### Credential check

`auth check` performs live, read-only validation calls to each provider's
API to verify credentials are valid without executing paid or state-modifying
operations.

### Storage

- **Registry**: `~/.config/open-llm-proxy/accounts.json` (0600) — metadata
  only (no secrets).
- **Secrets**: Per-account files under
  `~/.config/open-llm-proxy/accounts/<provider>/<name>.<ext>` (0600).
- **Override**: Set `OLP_CONFIG_DIR` to use a different base directory
  (useful for testing).

On first `auth` invocation, **existing credentials** are automatically
discovered and imported as `default` accounts — one-time, non-destructive,
per-provider. No credential is deleted or moved.

### Restart requirement

After changing credentials for an **api-key** provider (OpenRouter, NVIDIA)
stored in `~/.config/open-llm-proxy/env`, restart the proxy
(`open-llm-proxy restart`) so the updated env file is loaded. OAuth
provider credentials are read at runtime and take effect immediately.
