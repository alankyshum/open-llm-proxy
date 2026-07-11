from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import yaml


DEFAULT_CONFIG = Path.home() / ".config/open-llm-proxy/agent-config.yml"


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


def _service(args: argparse.Namespace) -> int:
    action = args.action
    target = _service_domain_target()

    if action == "status":
        result = _launchctl("print", target)
        if result.returncode != 0:
            print(f"Service '{SERVICE_LABEL}' is not loaded.")
            return 1
        running = _service_is_running()
        print(f"Service: {SERVICE_LABEL}")
        print(f"State:   {'running' if running else 'loaded (not running)'}")
        for line in result.stdout.splitlines():
            s = line.strip()
            if s.startswith(("pid =", "last exit code =", "state =")):
                print(f"  {s}")
        return 0 if running else 1

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
        # KeepAlive=true means the job restarts on plain `stop`; use `kill` to
        # signal the running process. The service remains loaded so `start`
        # (kickstart) can bring it back.
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
        ("status", "Show the launchd service state (pid, last exit)."),
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
