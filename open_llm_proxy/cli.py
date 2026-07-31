from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import yaml


DEFAULT_CONFIG = Path.home() / ".config/open-llm-proxy/agent-config.yml"

KNOWN_PROVIDERS = ("openrouter", "opencode", "github-copilot", "claude-cli", "nvidia")

# Provider metadata — single source of truth for both TUI and CLI branching.
# auth_kind: "api-key" (hidden prompt save) | "oauth-cli" (external login helper)
PROVIDERS: dict[str, dict[str, str]] = {
    "openrouter": {"label": "OpenRouter", "auth_kind": "api-key"},
    "nvidia": {"label": "NVIDIA (NIM)", "auth_kind": "api-key"},
    "opencode": {"label": "OpenCode", "auth_kind": "oauth-cli"},
    "github-copilot": {"label": "GitHub Copilot", "auth_kind": "oauth-cli"},
    "claude-cli": {"label": "Claude CLI", "auth_kind": "oauth-cli"},
}


def _help(args: argparse.Namespace) -> int:
    parser = build_parser()
    if args.topic is None:
        parser.print_help()
        return 0

    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    command_parser = subparsers.choices.get(args.topic)
    if command_parser is None:
        available = ", ".join(subparsers.choices)
        print(
            f"Unknown command: {args.topic}\nAvailable commands: {available}",
            file=sys.stderr,
        )
        return 2
    command_parser.print_help()
    return 0


def _serve(args: argparse.Namespace) -> int:
    from open_llm_proxy.server_launcher import launch_server

    launch_server(
        host=args.host,
        port=args.port,
        disable_admin_ui=args.disable_admin_ui,
        database_url=args.database_url,
        master_key=args.master_key,
        ui_username=args.ui_username,
        ui_password=args.ui_password,
        config_path=args.config,
    )
    return 0


def _ui(args: argparse.Namespace) -> int:
    import urllib.request
    import urllib.error
    import webbrowser

    url = args.url
    print(f"Checking LiteLLM Admin UI endpoint: {url}...")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "open-llm-proxy-cli/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.getcode()
            if status == 200:
                html = response.read().decode("utf-8", errors="ignore")
                if "Admin UI is disabled" in html:
                    print("Error: The server responded, but the Admin UI is disabled (middleware block).", file=sys.stderr)
                    return 1
                print("Success! LiteLLM Admin UI is running and responding with 200 OK.")
                if args.open:
                    print(f"Opening {url} in your default browser...")
                    webbrowser.open(url)
                return 0
            else:
                print(f"Error: Server responded with status code {status}.", file=sys.stderr)
                return 1
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"Error: Admin UI not found (404) at {url}. Is the UI disabled or port incorrect?", file=sys.stderr)
        else:
            print(f"Error: HTTP request failed with code {e.code}: {e.reason}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Error: Could not reach the server at {url}.", file=sys.stderr)
        print("Is the open-llm-proxy server running?", file=sys.stderr)
        print(f"Detail: {e.reason}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}", file=sys.stderr)
        return 1


SERVICE_LABEL = "com.user.open-llm-proxy"


def _launchctl(*cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["launchctl", *cmd], capture_output=True, text=True, check=False
    )


def _service_domain_target() -> str:
    import os

    return f"gui/{os.getuid()}/{SERVICE_LABEL}"


def _service_is_running() -> bool:
    result = _launchctl("print", _service_domain_target())
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        if "state = " in line:
            return "running" in line
    return False


def _service_is_ready() -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/healthz", timeout=2) as response:
            return response.getcode() == 200
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return False


def _service(args: argparse.Namespace) -> int:
    action = args.action
    target = _service_domain_target()

    if action == "status":
        result = _launchctl("print", target)
        if result.returncode != 0:
            print(f"Service '{SERVICE_LABEL}' is not loaded.")
            return 1
        running = _service_is_running()
        ready = running and _service_is_ready()
        if ready:
            state = "active"
        elif running:
            state = "starting (not ready)"
        else:
            state = "loaded (not running)"
        print(f"Service: {SERVICE_LABEL}")
        print(f"State:   {state}")
        print(f"Health:  {'ready' if ready else 'unavailable'}")
        for line in result.stdout.splitlines():
            s = line.strip()
            if s.startswith(("pid =", "last exit code =")):
                print(f"  {s}")
        return 0 if ready else 1

    if action == "start":
        result = _launchctl("kickstart", target)
        if result.returncode != 0:
            print(
                f"Error starting service: {result.stderr.strip() or result.stdout.strip()}",
                file=sys.stderr,
            )
            return 1
        print(f"Started {SERVICE_LABEL}.")
        return 0

    if action == "restart":
        result = _launchctl("kickstart", "-k", target)
        if result.returncode != 0:
            print(
                f"Error restarting service: {result.stderr.strip() or result.stdout.strip()}",
                file=sys.stderr,
            )
            return 1
        print(f"Restarted {SERVICE_LABEL}.")
        return 0

    if action == "stop":
        result = _launchctl("kill", "SIGTERM", target)
        if result.returncode != 0:
            print(
                f"Error stopping service: {result.stderr.strip() or result.stdout.strip()}",
                file=sys.stderr,
            )
            return 1
        print(f"Sent SIGTERM to {SERVICE_LABEL}.")
        return 0

    print(f"Unknown action: {action}", file=sys.stderr)
    return 2


def _check_openrouter() -> tuple[bool, str]:
    try:
        from open_llm_proxy import openrouter_creds
        key = openrouter_creds.get_persisted_api_key()
        if key and key.strip():
            return True, "credential discoverable"
    except Exception as e:
        return False, str(e)
    return False, "key is empty"


def _check_opencode() -> tuple[bool, str]:
    try:
        from open_llm_proxy import opencode_creds
        key = opencode_creds.get_opencode_api_key()
        if key and key.strip():
            return True, "credential discoverable"
    except Exception as e:
        return False, str(e)
    return False, "key is empty"


def _check_github_copilot() -> tuple[bool, str]:
    try:
        from open_llm_proxy import copilot_creds
        key = copilot_creds.get_oauth_token()
        if key and key.strip():
            return True, "credential discoverable"
    except Exception as e:
        return False, str(e)
    return False, "token is empty"


def _check_claude_cli() -> tuple[bool, str]:
    try:
        from open_llm_proxy import creds
        key = creds.get_api_key()
        if key and key.strip():
            return True, "credential discoverable"
    except Exception as e:
        return False, str(e)
    return False, "key is empty"


def _check_nvidia() -> tuple[bool, str]:
    try:
        from open_llm_proxy import nvidia_creds
        key = nvidia_creds.get_api_key()
        if key and key.strip():
            return True, "credential discoverable"
    except Exception as e:
        return False, str(e)
    return False, "key is empty"


def _run_opencode_login() -> int:
    try:
        res = subprocess.run(["opencode", "auth", "login", "https://opencode.ai"])
        if res.returncode != 0:
            print("Error: opencode auth login failed", file=sys.stderr)
            return res.returncode
    except FileNotFoundError:
        print("Error: opencode command not available", file=sys.stderr)
        return 127
    ok, msg = _check_opencode()
    if not ok:
        print(f"Error: OpenCode credential unresolved after authentication: {msg}", file=sys.stderr)
        return 1
    return 0


def _run_github_copilot_login() -> int:
    try:
        res = subprocess.run(
            [
                "opencode",
                "auth",
                "login",
                "--provider",
                "github-copilot",
                "--method",
                "Login with GitHub Copilot",
            ]
        )
        if res.returncode != 0:
            print("Error: github-copilot auth login failed", file=sys.stderr)
            return res.returncode
    except FileNotFoundError:
        print("Error: opencode command not available", file=sys.stderr)
        return 127
    ok, msg = _check_github_copilot()
    if not ok:
        print(f"Error: github-copilot credential unresolved after authentication: {msg}", file=sys.stderr)
        return 1
    return 0


def _run_claude_cli_login() -> int:
    try:
        res = subprocess.run(["claude", "auth", "login"])
        if res.returncode != 0:
            print("Error: claude auth login failed", file=sys.stderr)
            return res.returncode
    except FileNotFoundError:
        print("Error: claude command not available", file=sys.stderr)
        return 127
    ok, msg = _check_claude_cli()
    if not ok:
        print(f"Error: claude-cli credential unresolved after authentication: {msg}", file=sys.stderr)
        return 1
    return 0


def _ensure_registry_account(provider: str) -> None:
    """Add a @default account for *provider* if none exists in the registry.

    Does nothing if the provider already has accounts.  Uses the storage
    convention appropriate for each provider's legacy credential path.
    """
    from open_llm_proxy import account_registry

    if account_registry.list_accounts(provider):
        return
    if provider == "openrouter":
        account_registry.add_account(
            provider, storage="env-line", ref="OPENROUTER_API_KEY"
        )
    elif provider == "opencode":
        account_registry.add_account(provider, storage="external", ref="opencode")
    elif provider == "github-copilot":
        account_registry.add_account(provider, storage="external", ref="copilot")
    elif provider == "claude-cli":
        account_registry.add_account(provider, storage="external", ref="claude-default")
    elif provider == "nvidia":
        account_registry.add_account(
            provider, storage="env-line", ref="NVIDIA_API_KEY"
        )


def _capture_oauth_credential(provider: str) -> bytes | None:
    """Read the OAuth credential that was just established by an external login.

    Checks the legacy credential source for *provider* and returns the full
    credential payload as bytes, or ``None`` if nothing is discoverable.
    """
    import json

    if provider == "claude-cli":
        # ~/.claude/.credentials.json is written by `claude auth login`
        path = Path.home() / ".claude" / ".credentials.json"
        if path.is_file():
            try:
                data = json.loads(path.read_bytes())
                if isinstance(data, dict) and "claudeAiOauth" in data:
                    return json.dumps(data).encode()
            except Exception:
                pass
        return None

    if provider == "opencode":
        path = Path.home() / ".local" / "share" / "opencode" / "auth.json"
        if path.is_file():
            try:
                data = json.loads(path.read_bytes())
                if isinstance(data, dict) and "opencode" in data:
                    return json.dumps(data).encode()
            except Exception:
                pass
        return None

    if provider == "github-copilot":
        # Check opencode auth.json (github-copilot section) first
        path = Path.home() / ".local" / "share" / "opencode" / "auth.json"
        if path.is_file():
            try:
                data = json.loads(path.read_bytes())
                ghc = data.get("github-copilot")
                if isinstance(ghc, dict) and any(
                    ghc.get(k) for k in ("access", "refresh", "oauth_token")
                ):
                    return json.dumps({"github-copilot": ghc}).encode()
            except Exception:
                pass
        # Fall back to copilot.json
        fallback = Path.home() / ".config" / "open-llm-proxy" / "copilot.json"
        if fallback.is_file():
            try:
                return fallback.read_bytes()
            except Exception:
                pass
        return None

    return None


def add_provider_account(
    provider: str, name: str | None = None, key: str | None = None
) -> int:
    """Core logic to add a provider account shared by CLI and TUI.

    *provider* must be a key in PROVIDERS.
    If *name* is ``None`` and this is the first account, it becomes
    ``"default"``; if a second account, the caller must supply *name*.

    For ``api-key`` providers, *key* may be passed directly (TUI already
    captured it) or left as ``None`` (CLI reads from stdin or interactive
    prompt).

    For ``oauth-cli`` providers, *key* is ignored; the external login
    helper is run instead.

    Returns exit code (0 = success).
    """
    from open_llm_proxy.auth_migration import migrate_legacy_credentials
    from open_llm_proxy import account_registry

    migrate_legacy_credentials()

    existing = account_registry.list_accounts(provider)
    auth_kind = PROVIDERS[provider]["auth_kind"]

    # First account auto-named "default" if no name given.
    if not existing and name is None:
        name = "default"
    elif existing and name is None:
        print(
            f"Error: {provider} already has accounts; use --name to name the new account.",
            file=sys.stderr,
        )
        return 1
    if not existing:
        name = "default"

    if auth_kind == "api-key":
        if key is None:
            import getpass

            try:
                if not sys.stdin.isatty():
                    key = sys.stdin.read().strip()
                else:
                    key = getpass.getpass(
                        f"Enter {PROVIDERS[provider]['label']} API Key: "
                    ).strip()
            except Exception as e:
                print(f"Error reading API key: {e}", file=sys.stderr)
                return 1
        if not key:
            print("Error: API key cannot be empty", file=sys.stderr)
            return 1

        if name == "default":
            # Default account uses env-line storage
            if provider == "openrouter":
                from open_llm_proxy import openrouter_creds
                openrouter_creds.save_api_key(key)
            elif provider == "nvidia":
                from open_llm_proxy import nvidia_creds
                nvidia_creds.save_api_key(key)
            else:
                from open_llm_proxy import env_creds
                env_creds.set_env_key(f"{provider.upper()}_API_KEY", key)
            account_registry.add_account(
                provider, name, storage="env-line",
                ref=f"{provider.upper()}_API_KEY".replace("-", "_"),
            )
        else:
            # Named account: write key to per-account file
            account_registry.add_account(
                provider, name, storage="api-key", secret_bytes=key.encode()
            )
        print(f"Account {name!r} added for {provider}.")
        return 0

    elif auth_kind == "oauth-cli":
        if provider == "opencode":
            code = _run_opencode_login()
        elif provider == "github-copilot":
            code = _run_github_copilot_login()
        elif provider == "claude-cli":
            code = _run_claude_cli_login()
        else:
            return 2

        if code != 0:
            return code

        is_named = name is not None and name != "default"

        if is_named:
            # Snapshot the freshly-obtained credential into a per-account file
            cred_bytes = _capture_oauth_credential(provider)
            if cred_bytes is None:
                print(
                    f"Error: Could not capture OAuth credential after {provider} "
                    f"login for account {name!r}. No account was created.",
                    file=sys.stderr,
                )
                return 1

            storage_map = {
                "claude-cli": "claude-oauth",
                "opencode": "api-key",
                "github-copilot": "copilot-oauth",
            }
            account_registry.add_account(
                provider, name,
                storage=storage_map.get(provider, "api-key"),
                secret_bytes=cred_bytes,
            )
        else:
            ref_map = {
                "opencode": "opencode",
                "github-copilot": "copilot",
                "claude-cli": "claude-default",
            }
            account_registry.add_account(
                provider, name, storage="external", ref=ref_map[provider]
            )
        print(f"Account {name!r} added for {provider}.")
        return 0

    print(f"Error: unknown provider {provider!r}", file=sys.stderr)
    return 2


def _auth_set(args: argparse.Namespace) -> int:
    from open_llm_proxy.auth_migration import migrate_legacy_credentials

    migrate_legacy_credentials()

    import getpass
    from open_llm_proxy import openrouter_creds

    provider = args.provider
    if provider == "openrouter":
        try:
            if not sys.stdin.isatty():
                key = sys.stdin.read().strip()
            else:
                key = getpass.getpass("Enter OpenRouter API Key: ").strip()
            if not key:
                print("Error: API key cannot be empty", file=sys.stderr)
                return 1
            openrouter_creds.save_api_key(key)
            _ensure_registry_account("openrouter")
            print("Successfully saved OpenRouter API Key.")
            return 0
        except Exception as e:
            print(f"Error saving OpenRouter API Key: {e}", file=sys.stderr)
            return 1

    elif provider == "opencode":
        code = _run_opencode_login()
        if code == 0:
            _ensure_registry_account("opencode")
        return code

    elif provider == "github-copilot":
        code = _run_github_copilot_login()
        if code == 0:
            _ensure_registry_account("github-copilot")
        return code

    elif provider == "claude-cli":
        code = _run_claude_cli_login()
        if code == 0:
            _ensure_registry_account("claude-cli")
        return code

    elif provider == "nvidia":
        try:
            from open_llm_proxy import nvidia_creds
            if not sys.stdin.isatty():
                key = sys.stdin.read().strip()
            else:
                key = getpass.getpass("Enter NVIDIA API Key: ").strip()
            if not key:
                print("Error: API key cannot be empty", file=sys.stderr)
                return 1
            nvidia_creds.save_api_key(key)
            _ensure_registry_account("nvidia")
            print("Successfully saved NVIDIA API Key.")
            return 0
        except Exception as e:
            print(f"Error saving NVIDIA API Key: {e}", file=sys.stderr)
            return 1

    return 2


def _auth_accounts(args: argparse.Namespace) -> int:
    """List accounts per provider, marking the active one."""
    from open_llm_proxy.auth_migration import migrate_legacy_credentials
    from open_llm_proxy import account_registry

    migrate_legacy_credentials()

    providers = [args.provider] if args.provider else KNOWN_PROVIDERS
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
        print("No credentials configured.  Use `auth set <provider>` or `auth add <provider>`.")
    return 0


def _auth_add(args: argparse.Namespace) -> int:
    """Add a credential for a provider and register it as an account."""
    return add_provider_account(args.provider, name=args.name)


def _auth_rename(args: argparse.Namespace) -> int:
    """Rename an account (requires >=2 accounts for the provider)."""
    from open_llm_proxy.auth_migration import migrate_legacy_credentials
    from open_llm_proxy import account_registry

    migrate_legacy_credentials()

    try:
        account_registry.rename_account(args.provider, args.old, args.new)
        print(f"Account {args.old!r} renamed to {args.new!r} for {args.provider}.")
        return 0
    except account_registry.AccountRegistryError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _auth_use(args: argparse.Namespace) -> int:
    """Set the active account for a provider."""
    from open_llm_proxy.auth_migration import migrate_legacy_credentials
    from open_llm_proxy import account_registry

    migrate_legacy_credentials()

    try:
        account_registry.set_active(args.provider, args.name)
        # Best-effort invalidate the credential cache so the next untagged
        # call resolves fresh (defense in depth alongside the cache-key fix).
        try:
            from open_llm_proxy import creds as _creds
            _creds.clear_cache()
        except Exception:
            pass
        print(f"Active account for {args.provider} is now {args.name!r}.")
        return 0
    except account_registry.AccountRegistryError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _auth_remove(args: argparse.Namespace) -> int:
    """Remove an account (--force to remove the last one)."""
    from open_llm_proxy.auth_migration import migrate_legacy_credentials
    from open_llm_proxy import account_registry

    migrate_legacy_credentials()

    try:
        account_registry.remove_account(
            args.provider, args.name, force=args.force
        )
        print(f"Account {args.name!r} removed from {args.provider}.")
        return 0
    except account_registry.AccountRegistryError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _auth_check(args: argparse.Namespace) -> int:
    from open_llm_proxy.auth_migration import migrate_legacy_credentials
    from open_llm_proxy import connectivity

    migrate_legacy_credentials()

    providers_to_check = (
        [args.provider] if args.provider else list(KNOWN_PROVIDERS)
    )

    any_failed = False
    for p in providers_to_check:
        ok, msg = connectivity.check_provider(p)
        if ok:
            print(f"[OK] {p}: {msg}")
        else:
            print(f"[FAILED] {p}: {msg}", file=sys.stderr)
            any_failed = True

    return 1 if any_failed else 0


def _auth_orchestrator(args: argparse.Namespace) -> int:
    """Bare ``auth`` handler — launches TUI when interactive, else legacy flow.

    If ``--no-tui`` was passed or stdin is not a TTY, runs the original
    orchestrator that walks through each provider sequentially.
    """
    from open_llm_proxy.auth_migration import migrate_legacy_credentials

    migrate_legacy_credentials()
    # 1. OpenRouter
    ok, _ = _check_openrouter()
    if ok:
        print("[OK] openrouter: credential discoverable")
    else:
        from open_llm_proxy import openrouter_creds
        if not sys.stdin.isatty():
            key = sys.stdin.read().strip()
        else:
            try:
                key = openrouter_creds.get_api_key().strip()
            except Exception:
                import getpass
                try:
                    key = getpass.getpass("Enter OpenRouter API Key: ").strip()
                except Exception as e:
                    print(f"Error reading OpenRouter API Key: {e}", file=sys.stderr)
                    return 1
        if not key:
            print("Error: OpenRouter API key cannot be empty", file=sys.stderr)
            return 1
        try:
            openrouter_creds.save_api_key(key)
        except Exception as e:
            print(f"Error saving OpenRouter API Key: {e}", file=sys.stderr)
            return 1
        ok, msg = _check_openrouter()
        if not ok:
            print(f"Error: OpenRouter credential unresolved after saving: {msg}", file=sys.stderr)
            return 1
        print("[OK] openrouter: credential discoverable")

    # 2. OpenCode
    ok, _ = _check_opencode()
    if ok:
        print("[OK] opencode: credential discoverable")
    else:
        code = _run_opencode_login()
        if code != 0:
            return code
        print("[OK] opencode: credential discoverable")

    # 3. GitHub Copilot
    ok, _ = _check_github_copilot()
    if ok:
        print("[OK] github-copilot: credential discoverable")
    else:
        code = _run_github_copilot_login()
        if code != 0:
            return code
        print("[OK] github-copilot: credential discoverable")

    # 4. Claude CLI
    ok, _ = _check_claude_cli()
    if ok:
        print("[OK] claude-cli: credential discoverable")
    else:
        code = _run_claude_cli_login()
        if code != 0:
            return code
        print("[OK] claude-cli: credential discoverable")

    # 5. NVIDIA
    ok, _ = _check_nvidia()
    if ok:
        print("[OK] nvidia: credential discoverable")
    else:
        from open_llm_proxy import nvidia_creds
        if not sys.stdin.isatty():
            key = sys.stdin.read().strip()
        else:
            try:
                key = nvidia_creds.get_api_key().strip()
            except Exception:
                import getpass
                try:
                    key = getpass.getpass("Enter NVIDIA API Key: ").strip()
                except Exception as e:
                    print(f"Error reading NVIDIA API Key: {e}", file=sys.stderr)
                    return 1
        if not key:
            print("Error: NVIDIA API key cannot be empty", file=sys.stderr)
            return 1
        try:
            nvidia_creds.save_api_key(key)
        except Exception as e:
            print(f"Error saving NVIDIA API Key: {e}", file=sys.stderr)
            return 1
        ok, msg = _check_nvidia()
        if not ok:
            print(f"Error: NVIDIA credential unresolved after saving: {msg}", file=sys.stderr)
            return 1
        print("[OK] nvidia: credential discoverable")

    return 0


def _setup(args: argparse.Namespace) -> int:
    from open_llm_proxy.setup import configure

    interactive = not args.non_interactive and sys.stdin.isatty()
    configure(args.config, interactive=interactive, force=args.force)
    return 0


def _config(args: argparse.Namespace) -> int:
    from open_llm_proxy.config_gen import generate_config

    try:
        config = generate_config(str(args.config))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(config, indent=2))
    else:
        print(yaml.safe_dump(config, sort_keys=False), end="")
    return 0


def _reload(args: argparse.Namespace) -> int:
    """Hot-reload the live proxy's routing config without a restart.

    Atomically swaps the running Router's routes in-process (in-flight
    requests from other agents are preserved). Falls back to nothing —
    on failure the caller decides whether a disruptive restart is warranted.
    """
    import hashlib
    import time
    import urllib.error
    import urllib.request

    config_path = Path(args.config)
    try:
        expected_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    except OSError as exc:
        print(f"Error: cannot read {config_path}: {exc}", file=sys.stderr)
        return 1

    base = f"http://127.0.0.1:{args.port}"
    health_url = f"{base}/healthz"
    reload_url = f"{base}/internal/config/reload"
    timeout = args.timeout

    def _health() -> dict | None:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status != 200:
                    return None
                data = json.loads(response.read())
                return data if isinstance(data, dict) else None
        except Exception:
            return None

    health = _health()
    if not health or "config_hash" not in health:
        print(
            "Error: proxy is not reachable or lacks the reload contract "
            f"({health_url})",
            file=sys.stderr,
        )
        return 1
    if health.get("config_hash") == expected_hash:
        print("Proxy already has current routing config; nothing to reload.")
        return 0

    payload = json.dumps({"expected_hash": expected_hash}).encode()
    request = urllib.request.Request(
        reload_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read()).get("detail", "")
        except Exception:
            detail = getattr(exc, "reason", "") or str(exc)
        print(
            f"Error: proxy rejected reload (HTTP {exc.code}): {detail}",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        detail = getattr(exc, "reason", "") or str(exc)
        print(f"Error: proxy reload request failed: {detail}", file=sys.stderr)
        return 1

    if not isinstance(result, dict) or result.get("config_hash") != expected_hash:
        print("Error: proxy returned an unexpected config hash", file=sys.stderr)
        return 2

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        health = _health()
        if health and health.get("config_hash") == expected_hash:
            print("Reloaded: routing config applied without restart.")
            return 0
        time.sleep(0.25)

    print("Error: reload was not acknowledged within timeout", file=sys.stderr)
    return 2


def _models(args: argparse.Namespace) -> int:
    command = [args.opencode, "models"]
    if args.provider:
        command.append(args.provider)

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        print(
            f"Error: '{args.opencode}' was not found. Install OpenCode or pass "
            "--opencode /path/to/opencode.",
            file=sys.stderr,
        )
        return 1

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        print(f"Error: model discovery failed: {detail}", file=sys.stderr)
        return result.returncode

    models = list(dict.fromkeys(line.strip() for line in result.stdout.splitlines() if line.strip()))
    if args.search:
        needle = args.search.casefold()
        models = [model for model in models if needle in model.casefold()]

    if args.format == "json":
        print(json.dumps(models, indent=2))
    else:
        print("\n".join(models))
    return 0


def _handle_bare_auth(args: argparse.Namespace) -> int:
    """Dispatch bare ``auth``: TUI if TTY + not --no-tui, else orchestrator."""
    if sys.stdin.isatty() and not getattr(args, "no_tui", False):
        try:
            from open_llm_proxy.auth_tui import run_auth_tui
            return run_auth_tui()
        except ImportError:
            # questionary not installed — fall through to orchestrator
            pass
    return _auth_orchestrator(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open-llm-proxy",
        description="Run and configure the open-llm-proxy service.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    help_parser = subparsers.add_parser(
        "help",
        help="Show available commands or help for one command.",
    )
    help_parser.add_argument("topic", nargs="?", help="Command to describe.")
    help_parser.set_defaults(handler=_help)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the OpenAI-compatible proxy server.",
    )
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host address to bind (default: 0.0.0.0, reachable over Tailscale).")
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765).",
    )
    serve_parser.add_argument(
        "--config",
        "--config-path",
        dest="config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Agent configuration path (default: {DEFAULT_CONFIG}).",
    )
    serve_parser.add_argument(
        "--disable-admin-ui",
        action="store_true",
        help="Disable the LiteLLM Admin UI (run DB-less/Stateless).",
    )
    serve_parser.add_argument(
        "--database-url",
        help="PostgreSQL connection string for the Admin UI database.",
    )
    serve_parser.add_argument(
        "--master-key",
        help="The LITELLM_MASTER_KEY secret key to secure the proxy/UI.",
    )
    serve_parser.add_argument(
        "--ui-username",
        help="Login username for the LiteLLM Admin UI.",
    )
    serve_parser.add_argument(
        "--ui-password",
        help="Login password for the LiteLLM Admin UI.",
    )
    serve_parser.set_defaults(handler=_serve)

    ui_cmd_parser = subparsers.add_parser(
        "ui",
        help="Check status of the LiteLLM Admin UI and optionally open a browser.",
    )
    ui_cmd_parser.add_argument(
        "--url",
        default="http://127.0.0.1:8765/ui",
        help="The URL of the LiteLLM Admin UI (default: http://127.0.0.1:8765/ui).",
    )
    ui_cmd_parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open the UI URL in the default web browser if responsive.",
    )
    ui_cmd_parser.set_defaults(handler=_ui)

    for _action, _help_text in (
        ("start", "Start the open-llm-proxy launchd service."),
        ("stop", "Stop the running proxy process (service stays loaded)."),
        ("restart", "Restart the open-llm-proxy launchd service."),
        ("status", "Show service process and HTTP readiness state."),
    ):
        _p = subparsers.add_parser(_action, help=_help_text)
        _p.set_defaults(handler=_service, action=_action)

    setup_parser = subparsers.add_parser(
        "setup",
        help="Configure provider plans and rate-limit policy storage.",
    )
    setup_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Agent configuration path (default: {DEFAULT_CONFIG}).",
    )
    setup_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use configured/default plans without prompting.",
    )
    setup_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace provider plans already stored in SQLite.",
    )
    setup_parser.set_defaults(handler=_setup)

    config_parser = subparsers.add_parser(
        "config",
        aliases=["generate-config"],
        help="Generate the LiteLLM Router configuration.",
    )
    config_parser.add_argument(
        "--config",
        "--config-path",
        dest="config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Agent configuration path (default: {DEFAULT_CONFIG}).",
    )
    config_parser.add_argument(
        "--format",
        "--print",
        dest="format",
        choices=["yaml", "json"],
        default="yaml",
        help="Output format (default: yaml).",
    )
    config_parser.set_defaults(handler=_config)

    reload_parser = subparsers.add_parser(
        "reload",
        help="Hot-reload the live proxy's routing config without a restart.",
    )
    reload_parser.add_argument(
        "--config",
        "--config-path",
        dest="config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Deployed agent configuration path (default: {DEFAULT_CONFIG}).",
    )
    reload_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port the live proxy is listening on (default: 8765).",
    )
    reload_parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for the reload to be acknowledged (default: 15).",
    )
    reload_parser.set_defaults(handler=_reload)

    models_parser = subparsers.add_parser(
        "models",
        aliases=["available-models"],
        help="List exact model IDs available from OpenCode provider catalogs.",
    )
    models_parser.add_argument(
        "provider",
        nargs="?",
        help="Optional provider, such as openrouter, opencode, or github-copilot.",
    )
    models_parser.add_argument(
        "--search",
        metavar="TEXT",
        help="Only show model IDs containing TEXT (case-insensitive).",
    )
    models_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    models_parser.add_argument(
        "--opencode",
        default="opencode",
        help="OpenCode executable used for catalog discovery (default: opencode).",
    )
    models_parser.set_defaults(handler=_models)

    # Auth commands
    auth_parser = subparsers.add_parser(
        "auth",
        help="Manage provider credentials.",
    )
    auth_parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Skip interactive TUI even if stdin is a TTY.",
    )
    auth_parser.set_defaults(handler=_handle_bare_auth)
    auth_subparsers = auth_parser.add_subparsers(dest="subcommand", required=False)

    auth_set_parser = auth_subparsers.add_parser(
        "set",
        help="Set credential for a provider safely/interactively.",
    )
    auth_set_parser.add_argument(
        "provider",
        choices=KNOWN_PROVIDERS,
        help="Provider name to set.",
    )
    auth_set_parser.set_defaults(handler=_auth_set)

    auth_check_parser = auth_subparsers.add_parser(
        "check",
        help="Check provider connectivity using live API probes.",
    )
    auth_check_parser.add_argument(
        "provider",
        nargs="?",
        choices=KNOWN_PROVIDERS,
        help="Optional provider name to check.",
    )
    auth_check_parser.set_defaults(handler=_auth_check)

    # auth accounts [provider]
    auth_accounts_parser = auth_subparsers.add_parser(
        "accounts",
        help="List accounts per provider (or for one provider).",
    )
    auth_accounts_parser.add_argument(
        "provider",
        nargs="?",
        choices=KNOWN_PROVIDERS,
        help="Optional provider name to list.",
    )
    auth_accounts_parser.set_defaults(handler=_auth_accounts)

    # auth add <provider> [--name NAME]
    auth_add_parser = auth_subparsers.add_parser(
        "add",
        help="Add a credential for a provider and register it as an account.",
    )
    auth_add_parser.add_argument(
        "provider",
        choices=KNOWN_PROVIDERS,
        help="Provider name.",
    )
    auth_add_parser.add_argument(
        "--name",
        help="Account name (auto-named 'default' for the first account).",
    )
    auth_add_parser.set_defaults(handler=_auth_add)

    # auth rename <provider> <old> <new>
    auth_rename_parser = auth_subparsers.add_parser(
        "rename",
        help="Rename an account (requires >=2 accounts for the provider).",
    )
    auth_rename_parser.add_argument("provider", choices=KNOWN_PROVIDERS)
    auth_rename_parser.add_argument("old", help="Current account name.")
    auth_rename_parser.add_argument("new", help="New account name.")
    auth_rename_parser.set_defaults(handler=_auth_rename)

    # auth use <provider> <name>
    auth_use_parser = auth_subparsers.add_parser(
        "use",
        help="Set the active account for a provider.",
    )
    auth_use_parser.add_argument("provider", choices=KNOWN_PROVIDERS)
    auth_use_parser.add_argument("name", help="Account name to activate.")
    auth_use_parser.set_defaults(handler=_auth_use)

    # auth remove <provider> <name> [--force]
    auth_remove_parser = auth_subparsers.add_parser(
        "remove",
        help="Remove an account (--force to remove the last one).",
    )
    auth_remove_parser.add_argument("provider", choices=KNOWN_PROVIDERS)
    auth_remove_parser.add_argument("name", help="Account name to remove.")
    auth_remove_parser.add_argument(
        "--force",
        action="store_true",
        help="Allow removing the last account.",
    )
    auth_remove_parser.set_defaults(handler=_auth_remove)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
