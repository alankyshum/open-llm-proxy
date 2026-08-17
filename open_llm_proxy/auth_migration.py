from __future__ import annotations

import logging

log = logging.getLogger("open_llm_proxy.auth_migration")


def migrate_legacy_credentials() -> list[str]:
    """One-time non-destructive migration.

    For each known provider, IF the registry has no accounts yet AND a legacy
    credential is discoverable, import it as the @default account.

    Idempotent — safe to call repeatedly.  Returns list of providers migrated.
    One provider failure does not block others.
    """
    from open_llm_proxy import account_registry

    migrated: list[str] = []

    # ---- openrouter (env-line) ----
    try:
        if not account_registry.list_accounts("openrouter"):
            from open_llm_proxy import openrouter_creds

            try:
                key = openrouter_creds.get_persisted_api_key()
                if key and key.strip():
                    account_registry.add_account(
                        "openrouter",
                        "default",
                        storage="env-line",
                        ref="OPENROUTER_API_KEY",
                    )
                    migrated.append("openrouter")
            except Exception:  # intentional best-effort fallback or cleanup  # noqa: S110
                pass
    except Exception as e:
        log.warning("migrate: openrouter skipped (%s)", e)

    # ---- opencode (external, marker ref) ----
    try:
        if not account_registry.list_accounts("opencode"):
            from open_llm_proxy import opencode_creds

            try:
                key = opencode_creds.get_opencode_api_key()
                if key and key.strip():
                    account_registry.add_account(
                        "opencode", "default", storage="external", ref="opencode"
                    )
                    migrated.append("opencode")
            except Exception:  # intentional best-effort fallback or cleanup  # noqa: S110
                pass
    except Exception as e:
        log.warning("migrate: opencode skipped (%s)", e)

    # ---- github-copilot (external, marker ref) ----
    try:
        if not account_registry.list_accounts("github-copilot"):
            from open_llm_proxy import copilot_creds

            try:
                key = copilot_creds.get_oauth_token()
                if key and key.strip():
                    account_registry.add_account(
                        "github-copilot",
                        "default",
                        storage="external",
                        ref="copilot",
                    )
                    migrated.append("github-copilot")
            except Exception:  # intentional best-effort fallback or cleanup  # noqa: S110
                pass
    except Exception as e:
        log.warning("migrate: github-copilot skipped (%s)", e)

    # ---- claude-cli (external, marker ref) ----
    try:
        if not account_registry.list_accounts("claude-cli"):
            from open_llm_proxy import creds

            try:
                key = creds.get_api_key()
                if key and key.strip():
                    account_registry.add_account(
                        "claude-cli",
                        "default",
                        storage="external",
                        ref="claude-default",
                    )
                    migrated.append("claude-cli")
            except Exception:  # intentional best-effort fallback or cleanup  # noqa: S110
                pass
    except Exception as e:
        log.warning("migrate: claude-cli skipped (%s)", e)

    # ---- nvidia (env-line) ----
    try:
        if not account_registry.list_accounts("nvidia"):
            from open_llm_proxy import nvidia_creds

            try:
                key = nvidia_creds.get_api_key()
                if key and key.strip():
                    account_registry.add_account(
                        "nvidia",
                        "default",
                        storage="env-line",
                        ref="NVIDIA_API_KEY",
                    )
                    migrated.append("nvidia")
            except Exception:  # intentional best-effort fallback or cleanup  # noqa: S110
                pass
    except Exception as e:
        log.warning("migrate: nvidia skipped (%s)", e)

    return migrated
