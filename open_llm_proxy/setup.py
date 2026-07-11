from __future__ import annotations

import argparse
import sys
from pathlib import Path

from open_llm_proxy.config_gen import configured_model_tokens
from open_llm_proxy.rate_limit_catalog import DEFAULT_PLANS, PROVIDER_PLANS, get_plan_policy
from open_llm_proxy.rate_limit_state import RateLimitStore, load_rate_limit_policy


def _choose_plan(provider: str, default: str) -> str:
    plans = PROVIDER_PLANS[provider]
    names = list(plans)
    print(f"\nRate-limit plan for {provider}:")
    for index, name in enumerate(names, start=1):
        policy = plans[name]
        marker = " (default)" if name == default else ""
        print(f"  {index}. {name}: {policy.label}{marker}")
        print(f"     {policy.notes}")
    while True:
        answer = input(f"Select 1-{len(names)} [{names.index(default) + 1}]: ").strip()
        if not answer:
            return default
        try:
            selection = int(answer)
        except ValueError:
            print("Enter one of the listed numbers.")
            continue
        if 1 <= selection <= len(names):
            return names[selection - 1]
        print("Enter one of the listed numbers.")


def _print_inventory(store: RateLimitStore) -> None:
    rows = store.inventory()
    if not rows:
        print("No proxy models were found in the agent configuration.")
        return
    print("\nConfigured model rate-limit policies:")
    for row in rows:
        cooldown = row["default_cooldown_seconds"]
        print(
            f"  {row['provider']}/{row['model']}: "
            f"{row['plan']} ({row['label']}), fallback cooldown {cooldown}s"
        )
        print(
            f"    source: {row['source_url']} "
            f"(checked {row['source_checked_at']})"
        )


def configure(
    config_path: str | Path,
    *,
    interactive: bool,
    force: bool = False,
) -> RateLimitStore:
    policy_config = load_rate_limit_policy(config_path)
    configured_defaults = policy_config.get("configured_plans") or {}
    store = RateLimitStore(policy_config["database_path"], configured_plans=configured_defaults)
    model_keys = configured_model_tokens(config_path)
    store.register_models(model_keys)

    providers = sorted({key.split("/", 1)[0] for key in model_keys})
    for provider in providers:
        if provider not in PROVIDER_PLANS:
            print(f"Warning: no built-in rate-limit plans for {provider}", file=sys.stderr)
            continue
        if interactive:
            default = configured_defaults.get(provider, DEFAULT_PLANS[provider])
            plan = _choose_plan(provider, default)
            store._plans[provider] = (plan, get_plan_policy(provider, plan))

    _print_inventory(store)
    return store


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure provider plans and cache model rate-limit policies."
    )
    parser.add_argument(
        "--config",
        default=Path.home() / ".config/open-llm-proxy/agent-config.yml",
        type=Path,
        help="Path to agent-config.yml",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use configured/default plans without prompting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ask for and replace plans already stored in SQLite.",
    )
    args = parser.parse_args()
    interactive = not args.non_interactive and sys.stdin.isatty()
    configure(args.config, interactive=interactive, force=args.force)


if __name__ == "__main__":
    main()
