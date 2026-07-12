# Implementation Plan: Multi-Account Auth + Interactive TUI for `open-llm-proxy auth`

Status: COMPLETE — all phases (P1–P7) implemented, tested, and reviewed

## Implementation status

All phases delivered. Full suite: **379 passed, 5 failed** (the 5 are pre-existing live-network failures unrelated to this work: live Anthropic 429 ×4 + Prisma DB startup ×1).

- [x] **P1** — `account_registry.py`: unified registry (`accounts.json` 0600 + per-account secret files, atomic writes, `OLP_CONFIG_DIR` override, rename-guard ≥2 accounts).
- [x] **P2** — `creds.py` de-singletoned to per-account caches; named-account OAuth refresh isolated + atomic; legacy `@default` path unchanged; `anthropic_client.py` threads `account`.
- [x] **P3** — `provider@account/model` token grammar parsed/validated in `config_gen.py`; `provider_claude_cli.py` routes `claude_account`; chain alias rewrite preserves `@`.
- [x] **P4** — `rate_limit_state.py` `base_provider()` normalization: per-account cooldown buckets, shared plan/policy.
- [x] **P5** — `auth accounts | add | rename | use | remove | check` subcommands + idempotent non-destructive migration (`auth_migration.py`).
- [x] **P6** — interactive `questionary` TUI (`auth_tui.py`, base dep, `--no-tui`/non-TTY fallback); `nvidia` first-class api-key provider (`env_creds.py`, `nvidia_creds.py`, connectivity probe).
- [x] **P7** — README auth section rewritten.

### Deferred (by design — locked decision 3)

Only single-account (`@default`) is required end-to-end now. The storage/isolation layer already supports N accounts, but **named** (non-default) OAuth resolution for `github-copilot` and `opencode` is intentionally NOT wired end-to-end: `config_gen.map_token_to_deployment_params` raises a clear `ValueError("… not yet supported")` for a non-default `@account` on those providers rather than silently using the wrong account. `claude-cli`, `openrouter`, and `nvidia` support named accounts fully. Wire copilot/opencode named-account resolution when a second account for those providers is actually needed.

### Post-review hardening

Two review rounds (code-reviewer). Round 1 fixed 2 CRITICAL (named-claude add was non-resolvable; `auth use` was a no-op at resolution) + HIGH/MED/LOW. Round 2 confirmed resolved and fixed 2 residual should-fix: active-account cache staleness (now keyed by effective account + `clear_cache()` on `auth use`) and named-account env-key silent fallback (now fails closed with `RuntimeError`).

Scope: two independent-but-related features in `open_llm_proxy/`

1. Feature A — multiple named accounts per provider (starting with `claude-cli`), surfaced through `open-llm-proxy auth`, with zero path knowledge required from the user.
2. Feature B — an interactive TUI for `open-llm-proxy auth` (à la `opencode auth`): pick a provider from a list → choose "paste API key" or "OAuth login". This replaces the ad-hoc `nvidia_nim` wiring that today only exists at the code/agent-config level.

---

## 0. Current-state findings (grounding)

- `cli.py` holds all auth UX: `_auth_orchestrator` (bare `auth`), `_auth_set` (`auth set <provider>`), `_auth_check` (`auth check`). Provider list hard-coded to `["openrouter","opencode","github-copilot","claude-cli"]` in three places (argparse choices + orchestrator).
- Per-provider credential modules, each with its own storage convention:
  - `openrouter_creds.py` → `OPENROUTER_API_KEY` line in `~/.config/open-llm-proxy/env` (atomic rewrite, 0600).
  - `opencode_creds.py` → reads `~/.local/share/opencode/auth.json` (written by `opencode auth login`).
  - `copilot_creds.py` → keychain / `secret-tool` / opencode auth.json / `~/.config/open-llm-proxy/copilot.json` fallback; OAuth device-flow constants present (`COPILOT_CLIENT_ID`).
  - `creds.py` (Anthropic/claude-cli) → **process-global singleton** (`_cached_key`, `_in_memory_oauth`, module-level TTL cache). Resolution order: env → macOS keychain (`"Claude Code-credentials"`, account = `$USER`) → `~/.claude.json` → `~/.claude/.credentials.json`. OAuth refresh via `refresh_anthropic_oauth()` writes back to the fixed `~/.claude/.credentials.json` + keychain.
- `connectivity.py::check_provider(provider)` does a live read-only probe per provider; single credential per provider assumed.
- Config/routing path for a chain token `claude-cli/<model>`:
  - `config_gen.py::map_token_to_deployment_params` emits `{"model":"claude-cli/<rest>"}` (no credential/account info).
  - `provider_claude_cli.py::ClaudeCliLLM` calls `anthropic_client.stream_messages/send_messages` which calls `creds.get_api_key()` (singleton) — never reads `litellm_params`.
  - `rate_limit_key = token` string (e.g. `claude-cli/claude-opus-4-8`) → one cooldown bucket per provider+model, keyed `provider/model` split on first `/` in `rate_limit_state.py`.
- `nvidia_nim` today: NO dedicated code. Works only because `map_token_to_deployment_params` `else`-branch passes the token through to LiteLLM's native `nvidia_nim` provider, which reads `NVIDIA_API_KEY`/`NVIDIA_NIM_API_KEY` from env. There is no `auth set nvidia` path — the user sets the env var manually. **Feature B fixes this.**
- Deps (`pyproject.toml`): `litellm[proxy]`, `prisma`, `httpx[http2]`, `pyyaml`. Python `>=3.10`. No TUI lib yet.

---

## 1. Design decisions

### 1.1 Account model (Feature A)

- Every provider credential lives under a **named account**. The first credential a user adds for a provider is auto-named `@default`. Accounts are only *renameable* once a **second** account exists for that provider (matches the user's requirement: "when user add another key, only then let user rename the accounts, including the first one").
- Account naming rules: `[a-z0-9][a-z0-9_-]*`, lowercased, max 32 chars, unique per provider. The literal `default` is reserved as the initial name but may be renamed later.
- **Storage** — a single new JSON registry owned by us, so the user never sees paths:
  `~/.config/open-llm-proxy/accounts.json` (0600), shape:
  ```json
  {
    "version": 1,
    "providers": {
      "claude-cli": {
        "active": "default",
        "accounts": {
          "default": { "created_at": "...", "storage": "claude-oauth", "ref": "claude-cli/default.credentials.json" },
          "work":    { "created_at": "...", "storage": "claude-oauth", "ref": "claude-cli/work.credentials.json" }
        }
      },
      "openrouter": {
        "active": "default",
        "accounts": { "default": { "storage": "env-line", "ref": "OPENROUTER_API_KEY" } }
      }
    }
  }
  ```
- Per-account **secret material** is stored in a per-account file under `~/.config/open-llm-proxy/accounts/<provider>/<account>.<ext>` (0600). The registry only holds metadata + a `ref`. This keeps OAuth JSON (claude), raw API keys (openrouter/nvidia), and device-flow tokens (copilot) uniformly addressable without the user knowing paths.
- **Backward compatibility**: on first run, a migration seeds `@default` for each provider that already has a discoverable credential using the *existing* resolution (e.g. claude keychain/`~/.claude/.credentials.json`, `OPENROUTER_API_KEY` env line). Existing single-account behavior is preserved when `accounts.json` is absent (`account=None` everywhere resolves to legacy path).

### 1.2 Token / chain syntax (Feature A wiring)

- Extend the chain token grammar with an optional `@account` suffix on the **provider**:
  `claude-cli@work/claude-opus-4-8`. Absent `@account` ⇒ the provider's `active` account (default `@default`).
- This makes each account a distinct deployment ⇒ **free independent rate-limit buckets and failover** between two Claude accounts, e.g.:
  ```yaml
  model: "open-llm-proxy/[claude-cli@work/claude-opus-4-8,claude-cli@home/claude-opus-4-8,github-copilot/claude-opus-4.8]"
  ```

### 1.3 TUI library choice (Feature B)

- Recommend **`questionary`** (MIT, built on `prompt_toolkit`). Rationale: tiny, purpose-built for exactly this (interactive `select`, `text`, `password`, `confirm`), cross-platform, no async requirement, degrades to non-TTY safely. Alternatives considered:
  - `InquirerPy` — also good (fork of PyInquirer), slightly larger API; acceptable second choice.
  - `textual` / full TUI — overkill for a linear pick→prompt flow.
  - `rich` alone — great rendering but no interactive selection primitive.
- Add `questionary` as an **optional dependency** group `tui` and import lazily inside the auth command so the proxy server path never imports prompt_toolkit. If `questionary` is missing or stdin is not a TTY, fall back to the current numbered-input prompts (already the style used in `setup.py::_choose_plan`).

---

## 2. Delegation / phase table

| Phase | Title | Files | Depends on |
|-------|-------|-------|------------|
| P1 | Account registry core | `account_registry.py` (new), tests | — |
| P2 | claude-cli per-account creds (de-singleton) | `creds.py`, `anthropic_client.py`, tests | P1 |
| P3 | Token `@account` parsing + routing | `config_gen.py`, `provider_claude_cli.py`, `model_chain_middleware.py`, tests | P1,P2 |
| P4 | Rate-limit per-account keys | `rate_limit_state.py`, `rate_limit_catalog.py`, tests | P3 |
| P5 | `auth` account subcommands (add/list/rename/use/remove) | `cli.py`, `connectivity.py`, tests | P1–P4 |
| P6 | Interactive TUI (`questionary`) + nvidia provider registration | `cli.py`, `auth_tui.py` (new), `pyproject.toml`, tests | P5 |
| P7 | Docs + README | `README.md` | P5,P6 |

---

## 3. Phase details

### P1 — Account registry core (`account_registry.py`)
- New module. Pure storage/metadata, no network. Public API:
  - `list_providers() -> list[str]`
  - `list_accounts(provider) -> list[AccountInfo]`
  - `active_account(provider) -> str`
  - `add_account(provider, name=None, *, storage, secret_writer) -> AccountInfo` (auto-names `default` when first; enforces naming rules & uniqueness)
  - `rename_account(provider, old, new)` — **guard**: raises if provider has `<2` accounts (enforces the "rename only after a 2nd key" rule).
  - `set_active(provider, name)`, `remove_account(provider, name)` (re-points `active` if needed; forbids removing last account without confirmation flag).
  - `resolve_secret_ref(provider, name) -> Path`
- Atomic write of `accounts.json` (mkstemp + `replace`, chmod 0600) mirroring `openrouter_creds.save_api_key`.
- Migration helper `seed_from_legacy()` invoked lazily on first registry load.
- Tests: naming validation, default auto-naming, rename-guard (<2 accounts → error; ≥2 → ok), atomic write perms, migration seeds `@default`.

### P2 — claude-cli per-account credentials
- Refactor `creds.py`: replace module-global `_cached_key`/`_in_memory_oauth` singletons with a **per-account cache dict** keyed by account name. Signatures gain `account: str | None = None` (None ⇒ active/`default`, preserving legacy resolution when registry absent):
  - `get_api_key(account=None)`, `refresh_anthropic_oauth(stale_token=None, account=None)`, `clear_cache(account=None)`, `reset_oauth_state(account=None)`.
- For a named non-default account, source/refresh/write-back the OAuth JSON at the registry's per-account path instead of the fixed `~/.claude/.credentials.json` + keychain. `account=None`/`default` keeps today's keychain+`~/.claude` behavior for backward compat.
- `anthropic_client.py`: thread `account` through `stream_messages`, `send_messages`, `fetch_models` → `creds.get_api_key(account=...)` and refresh calls.
- Tests: two accounts resolve to different keys; refresh writes to the correct per-account file; legacy `account=None` still reads keychain/`~/.claude`.

### P3 — Token `@account` parsing + routing
- `config_gen.py`:
  - `parse_fallback_chain`: accept `provider@account/model`; validate account chars; keep `/` requirement.
  - `map_token_to_deployment_params`: split optional `@account`; for `claude-cli` add `litellm_params["claude_account"] = account` (omit when default). Normalize provider-prefix checks to handle the `@` (use base provider before `@`).
  - Deployment alias + `rate_limit_key` must include the account tag (e.g. `claude-cli@work/claude-opus-4-8`) so buckets stay distinct.
- `provider_claude_cli.py`: read `claude_account` from `kwargs`/`optional_params`/`litellm_params`; pass to `anthropic_client`; include account in `rate_limit_origin_key`.
- `model_chain_middleware.py`: the comma→semicolon alias rewrite must preserve `@account` tokens unchanged (they contain no comma; verify the `@` survives the alias replace).
- Tests mirror `test_config_gen.py`: chain with two `claude-cli@x` entries → two deployments, two rate_limit_keys, fallback mapping intact.

### P4 — Rate-limit per-account keys
- `rate_limit_state.py`: `register_models` / `_rate_limit_key` split on first `/` still works, but provider column becomes `claude-cli@work`. Add a `base_provider()` normalizer (`split("@")[0]`) used wherever a **plan/policy** lookup happens so `claude-cli@work` and `claude-cli@home` both map to the `claude-cli` plan, while keeping **separate cooldown rows** per account.
- `rate_limit_catalog.py` / `_rate_limit_key_for_exception` `accepted_providers`: match on base provider.
- Tests: two Claude accounts get independent `retry_at` rows; both resolve `claude-cli` plan policy; 429 on one does not cool down the other.

### P5 — `auth` account subcommands
- New nested commands under `auth`:
  - `auth accounts <provider>` — list accounts (mark active).
  - `auth add <provider> [--name N]` — add credential (paste key or OAuth per provider); first one silently `default`.
  - `auth rename <provider> <old> <new>` — guarded (needs ≥2 accounts).
  - `auth use <provider> <name>` — set active.
  - `auth remove <provider> <name>`.
- Refactor `_auth_set`/`_auth_orchestrator` to go through `account_registry` + per-provider secret writers. Keep existing top-level `auth`/`auth set`/`auth check` behavior working (operating on the active account).
- `connectivity.check_provider(provider, account=None)` gains optional account.
- Tests: add→default, add 2nd→both listed, rename first after 2nd exists, `use` switches active, `check` probes active account.

### P6 — Interactive TUI + nvidia registration
- Add `questionary` to `[project.optional-dependencies].tui` and (optionally) the default deps if we want `auth` interactive out-of-the-box. Lazy-import inside `auth_tui.py`.
- New `auth_tui.py`: `run_auth_tui()` —
  1. `questionary.select` provider from a **registry-driven** list (now including `nvidia` and any future provider via a single `PROVIDERS` table with metadata: display name, auth kind = `api_key` | `oauth-cli` | `oauth-device`).
  2. Branch: `api_key` → `questionary.password` (hidden) → save; `oauth-*` → run existing login helper (`opencode`/`claude`/copilot device flow).
  3. Offer account naming only when a 2nd account is being added (reuse P1 rule).
- `nvidia`: add a tiny `nvidia_creds.py` writing `NVIDIA_API_KEY` into `~/.config/open-llm-proxy/env` via the same env-line writer as openrouter (generalize `openrouter_creds`'s writer into a shared `_env_creds.set_env_key(name, value)` helper), plus a `connectivity` probe (`GET https://integrate.api.nvidia.com/v1/models`). Register `nvidia` in the `PROVIDERS` table and argparse choices.
- Bare `open-llm-proxy auth` with a TTY → launches the TUI; non-TTY or `--no-tui` → current orchestrator behavior.
- Tests: PROVIDERS table drives choices; non-TTY fallback path; nvidia key save + env line; questionary calls mocked.

### P7 — Docs
- README: document `@account` chain syntax, `auth add/rename/use/remove`, the interactive TUI, and nvidia setup. Note the two-Claude-accounts failover example.

---

## 4. Locked decisions

1. **Registry scope**: UNIFIED across all providers. `accounts.json` + per-account secret files become the single source of truth for `openrouter`, `opencode`, `github-copilot`, `claude-cli`, and `nvidia`. A one-time migration MUST import every existing discoverable credential (openrouter env line, opencode auth.json, copilot keychain/fallback, claude keychain/`~/.claude/.credentials.json`) as that provider's `@default` account. Migration is idempotent and non-destructive (legacy sources are read, not deleted).
2. **`questionary`**: a BASE dependency. `open-llm-proxy auth` is interactive by default when stdin is a TTY; `--no-tui` and non-TTY fall back to numbered prompts.
3. **Multi-account isolation for ALL OAuth providers**: the same per-account snapshot pattern used for `claude-cli` applies to `github-copilot` and any other OAuth provider — after each login the fresh credential is persisted into an isolated per-account file so a subsequent login for a different account cannot overwrite it. NOTE: only single-account (`@default`) needs to actually work end-to-end now (user has no 2nd account yet), but the storage/isolation design MUST already support N accounts. Users can delete an authed account or re-auth at any time.
4. **Account tag separator**: `@` CONFIRMED. No catalog model IDs use `@`.

---

## 5. Test/verification commands

- `pytest tests/ -q` (add: `test_account_registry.py`, `test_creds_multi_account.py`, `test_config_gen.py` additions, `test_rate_limit_state.py` additions, `test_auth_tui.py`).
- Manual: `open-llm-proxy auth` (TUI), `open-llm-proxy auth add claude-cli --name work`, `open-llm-proxy auth accounts claude-cli`, `open-llm-proxy config` shows two `claude-cli@*` deployments, `open-llm-proxy auth check claude-cli`.
