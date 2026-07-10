from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SOURCE_CHECKED_AT = "2026-07-10"


@dataclass(frozen=True)
class PlanPolicy:
    label: str
    default_cooldown_seconds: int
    quota_limited: bool
    limits: dict[str, Any]
    source_url: str
    notes: str


CLAUDE_SOURCE = "https://support.claude.com/en/articles/9797557-usage-limit-best-practices"
COPILOT_SOURCE = "https://docs.github.com/en/copilot/get-started/plans"
GOOGLE_SOURCE = "https://ai.google.dev/gemini-api/docs/rate-limits"
OPENROUTER_SOURCE = "https://openrouter.ai/docs/api/reference/limits"
OPENCODE_SOURCE = "https://opencode.ai/docs/zen"


def _claude_plan(label: str) -> PlanPolicy:
    return PlanPolicy(
        label=label,
        default_cooldown_seconds=5 * 60 * 60,
        quota_limited=True,
        limits={"session_window_seconds": 18_000, "weekly_limit": True},
        source_url=CLAUDE_SOURCE,
        notes="Claude subscription usage has a five-hour session window and a weekly limit.",
    )


def _transient_plan(
    label: str,
    source_url: str,
    *,
    quota_limited: bool,
    limits: dict[str, Any],
    notes: str,
) -> PlanPolicy:
    return PlanPolicy(
        label=label,
        default_cooldown_seconds=60,
        quota_limited=quota_limited,
        limits=limits,
        source_url=source_url,
        notes=notes,
    )


PROVIDER_PLANS: dict[str, dict[str, PlanPolicy]] = {
    "claude-cli": {
        "free": _claude_plan("Free"),
        "pro": _claude_plan("Pro"),
        "max-5x": _claude_plan("Max 5x"),
        "max-20x": _claude_plan("Max 20x"),
        "api": _transient_plan(
            "API",
            "https://docs.anthropic.com/en/api/rate-limits",
            quota_limited=True,
            limits={"dimensions": ["rpm", "tpm", "input_tokens", "output_tokens"]},
            notes="API limits vary by organization tier; provider reset metadata is authoritative.",
        ),
    },
    "github-copilot": {
        plan: _transient_plan(
            label,
            COPILOT_SOURCE,
            quota_limited=plan != "unlimited",
            limits={"billing": "plan allowance", "transport_rate_limits": True},
            notes=(
                "No quota cooldown is imposed by the proxy; provider reset metadata "
                "is used, with a one-minute transient fallback."
            ),
        )
        for plan, label in {
            "free": "Free",
            "pro": "Pro",
            "pro-plus": "Pro+",
            "max": "Max",
            "business": "Business",
            "enterprise": "Enterprise",
            "unlimited": "Unlimited tokens",
        }.items()
    },
    "google": {
        plan: _transient_plan(
            label,
            GOOGLE_SOURCE,
            quota_limited=True,
            limits={
                "scope": "project",
                "dimensions": ["rpm", "tpm", "rpd"],
                "daily_reset": "midnight Pacific",
                "live_limits_url": "https://aistudio.google.com/rate-limit",
            },
            notes=(
                "Gemini limits vary by model, project, and account status. "
                "AI Studio and provider reset metadata are authoritative."
            ),
        )
        for plan, label in {
            "free": "Free tier",
            "tier-1": "Paid tier 1",
            "tier-2": "Paid tier 2",
            "tier-3": "Paid tier 3",
        }.items()
    },
    "openrouter": {
        "free": _transient_plan(
            "Free, under $10 credits purchased",
            OPENROUTER_SOURCE,
            quota_limited=True,
            limits={"rpm_free_models": 20, "rpd_free_models": 50},
            notes="Limits apply to model IDs ending in :free.",
        ),
        "free-with-credits": _transient_plan(
            "Free models, at least $10 credits purchased",
            OPENROUTER_SOURCE,
            quota_limited=True,
            limits={"rpm_free_models": 20, "rpd_free_models": 1000},
            notes="Limits apply to model IDs ending in :free.",
        ),
        "payg": _transient_plan(
            "Pay as you go",
            OPENROUTER_SOURCE,
            quota_limited=False,
            limits={"provider_capacity_limits": True},
            notes="Paid models have no OpenRouter account RPM quota; upstream capacity still applies.",
        ),
    },
    "opencode": {
        "free-models": _transient_plan(
            "Zen free models",
            OPENCODE_SOURCE,
            quota_limited=True,
            limits={"published_numeric_limit": False},
            notes="Zen publishes free model pricing but no fixed numeric request quota.",
        ),
        "payg": _transient_plan(
            "Zen pay as you go",
            OPENCODE_SOURCE,
            quota_limited=False,
            limits={"published_numeric_limit": False},
            notes="Zen publishes pricing but no fixed numeric request quota.",
        ),
    },
}

DEFAULT_PLANS = {
    "claude-cli": "free",
    "github-copilot": "free",
    "google": "free",
    "openrouter": "free",
    "opencode": "free-models",
}


def get_plan_policy(provider: str, plan: str) -> PlanPolicy:
    try:
        return PROVIDER_PLANS[provider][plan]
    except KeyError as exc:
        available = ", ".join(PROVIDER_PLANS.get(provider, {}))
        raise ValueError(
            f"unknown plan {plan!r} for {provider}; available plans: {available}"
        ) from exc
