from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import questionary


def run_auth_tui() -> int:
    """Interactive TUI for ``open-llm-proxy auth``.

    Lazy-imports ``questionary`` so module-level import never pulls in
    ``prompt_toolkit``.  Returns 0 on clean exit, non-zero on error.

    If stdin is not a TTY, returns 127 so the caller can fall back to
    the non-interactive path.
    """
    if not sys.stdin.isatty():
        return 127

    # Lazy import: questionary is a dependency but we still import inside
    # the function so that the proxy server path never loads prompt_toolkit.
    import questionary

    from open_llm_proxy.auth_migration import migrate_legacy_credentials
    from open_llm_proxy import account_registry
    from open_llm_proxy import cli
    from open_llm_proxy import connectivity

    migrate_legacy_credentials()

    while True:
        action = _ask_action(questionary)
        if action is None:  # Ctrl-C
            return 0

        if action == "quit":
            return 0

        if action == "add":
            rc = _tui_add(questionary)
            if rc != 0:
                return rc
            continue

        if action == "list":
            _tui_list()
            continue

        if action == "switch":
            rc = _tui_switch(questionary)
            if rc != 0:
                return rc
            continue

        if action == "rename":
            rc = _tui_rename(questionary)
            if rc != 0:
                return rc
            continue

        if action == "remove":
            rc = _tui_remove(questionary)
            if rc != 0:
                return rc
            continue

        if action == "check":
            _tui_check()
            continue


def _ask_action(q: questionary) -> str | None:
    """Return the selected action key, or None on Ctrl-C."""
    try:
        result = q.select(
            "What would you like to do?",
            choices=[
                q.Choice(title="Add / re-auth a provider", value="add"),
                q.Choice(title="List accounts", value="list"),
                q.Choice(title="Switch active account", value="switch"),
                q.Choice(title="Rename account", value="rename"),
                q.Choice(title="Remove account", value="remove"),
                q.Choice(title="Check connectivity", value="check"),
                q.Choice(title="Quit", value="quit"),
            ],
        ).ask()
        return result
    except KeyboardInterrupt:
        return None


def _tui_add(q: questionary) -> int:
    """Interactively add a provider account."""
    from open_llm_proxy import cli as _cli

    providers = _cli.PROVIDERS
    provider_keys = sorted(providers.keys())

    try:
        provider = q.select(
            "Select a provider:",
            choices=[q.Choice(title=providers[k]["label"], value=k) for k in provider_keys],
        ).ask()
    except KeyboardInterrupt:
        return 0
    if provider is None:
        return 0

    from open_llm_proxy import account_registry

    existing = account_registry.list_accounts(provider)

    name = "default"
    if existing:
        try:
            name = q.text(
                f"Account name for this {providers[provider]['label']} credential:",
                validate=lambda val: bool(
                    val.strip()
                    and account_registry.normalize_account_name(val.strip())
                ),
            ).ask()
        except KeyboardInterrupt:
            return 0
        if name is None:
            return 0
        name = name.strip()

    auth_kind = providers[provider]["auth_kind"]

    if auth_kind == "api-key":
        try:
            secret = q.password(f"Enter {providers[provider]['label']} API Key:").ask()
        except KeyboardInterrupt:
            return 0
        if secret is None or not secret.strip():
            print("Error: API key cannot be empty", file=sys.stderr)
            return 1

        # Reuse the shared add_provider_account logic
        return _cli.add_provider_account(provider, name=name, key=secret.strip())

    elif auth_kind == "oauth-cli":
        try:
            confirmed = q.confirm(
                f"Run {providers[provider]['label']} login helper now?"
            ).ask()
        except KeyboardInterrupt:
            return 0
        if confirmed is None or not confirmed:
            return 0

        return _cli.add_provider_account(provider, name=name)

    return 2


def _tui_list() -> None:
    """List all accounts."""
    from open_llm_proxy import account_registry
    from open_llm_proxy import cli as _cli

    providers = _cli.KNOWN_PROVIDERS
    any_found = False

    for p in providers:
        accounts = account_registry.list_accounts(p)
        if not accounts:
            continue
        any_found = True
        print(f"{p}:")
        for a in accounts:
            prefix = "  * " if a.is_active else "    "
            print(f"{prefix}{a.name}")

    if not any_found:
        print("No credentials configured.  Use the Add option to add one.")


def _tui_switch(q: questionary) -> int:
    """Switch the active account for a provider."""
    from open_llm_proxy import account_registry
    from open_llm_proxy import cli as _cli

    providers = _cli.KNOWN_PROVIDERS

    # Only show providers that have accounts
    viable = [p for p in providers if account_registry.list_accounts(p)]
    if not viable:
        print("No accounts to switch.  Add one first.", file=sys.stderr)
        return 1

    try:
        provider = q.select(
            "Provider:", choices=viable
        ).ask()
    except KeyboardInterrupt:
        return 0
    if provider is None:
        return 0

    accounts = account_registry.list_accounts(provider)
    try:
        name = q.select(
            f"Account to activate for {provider}:",
            choices=[a.name for a in accounts],
        ).ask()
    except KeyboardInterrupt:
        return 0
    if name is None:
        return 0

    try:
        account_registry.set_active(provider, name)
        print(f"Active account for {provider} is now {name!r}.")
        return 0
    except account_registry.AccountRegistryError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _tui_rename(q: questionary) -> int:
    """Rename an account for a provider."""
    from open_llm_proxy import account_registry
    from open_llm_proxy import cli as _cli

    providers = _cli.KNOWN_PROVIDERS
    viable = [p for p in providers if len(account_registry.list_accounts(p)) >= 2]
    if not viable:
        print(
            "No provider has multiple accounts yet.  "
            "Add a second account before renaming.",
            file=sys.stderr,
        )
        return 1

    try:
        provider = q.select("Provider:", choices=viable).ask()
    except KeyboardInterrupt:
        return 0
    if provider is None:
        return 0

    accounts = account_registry.list_accounts(provider)
    try:
        old = q.select("Account to rename:", choices=[a.name for a in accounts]).ask()
    except KeyboardInterrupt:
        return 0
    if old is None:
        return 0

    try:
        new = q.text("New name:", validate=lambda val: bool(
            val.strip()
            and account_registry.normalize_account_name(val.strip())
        )).ask()
    except KeyboardInterrupt:
        return 0
    if new is None or not new.strip():
        return 0
    new = new.strip()

    try:
        account_registry.rename_account(provider, old, new)
        print(f"Account {old!r} renamed to {new!r} for {provider}.")
        return 0
    except account_registry.AccountRegistryError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _tui_remove(q: questionary) -> int:
    """Remove an account for a provider."""
    from open_llm_proxy import account_registry
    from open_llm_proxy import cli as _cli

    providers = _cli.KNOWN_PROVIDERS
    viable = [p for p in providers if account_registry.list_accounts(p)]
    if not viable:
        print("No accounts to remove.", file=sys.stderr)
        return 1

    try:
        provider = q.select("Provider:", choices=viable).ask()
    except KeyboardInterrupt:
        return 0
    if provider is None:
        return 0

    accounts = account_registry.list_accounts(provider)
    try:
        name = q.select("Account to remove:", choices=[a.name for a in accounts]).ask()
    except KeyboardInterrupt:
        return 0
    if name is None:
        return 0

    force = False
    if len(accounts) == 1:
        try:
            force = q.confirm(
                "This is the last account.  Remove it?",
                default=False,
            ).ask()
        except KeyboardInterrupt:
            return 0
        if force is None:
            return 0

    try:
        account_registry.remove_account(provider, name, force=force)
        print(f"Account {name!r} removed from {provider}.")
        return 0
    except account_registry.AccountRegistryError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _tui_check() -> None:
    """Check connectivity for all providers."""
    from open_llm_proxy import cli as _cli
    from open_llm_proxy import connectivity

    for p in _cli.KNOWN_PROVIDERS:
        ok, msg = connectivity.check_provider(p)
        if ok:
            print(f"[OK] {p}: {msg}")
        else:
            print(f"[FAILED] {p}: {msg}", file=sys.stderr)
