from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import yaml


DEFAULT_CONFIG = Path.home() / ".config/kilo-claude-proxy/agent-config.yml"


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

    launch_server(host=args.host, port=args.port)
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
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host address to bind.")
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765).",
    )
    serve_parser.set_defaults(handler=_serve)

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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
