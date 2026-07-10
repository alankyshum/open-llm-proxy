import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from open_llm_proxy.rate_limit_state import (
    PersistentRateLimitCallback,
    RateLimitStore,
    load_rate_limit_policy,
)


class RateLimitedError(Exception):
    status_code = 429

    def __init__(self, headers=None):
        super().__init__("rate limited")
        self.headers = headers or {}


class WrappedProviderError(Exception):
    status_code = 503

    def __init__(self, original_exception):
        super().__init__("provider stream failed")
        self.original_exception = original_exception


def deployment(key):
    return {"model_info": {"rate_limit_key": key}}


def test_load_rate_limit_policy(tmp_path):
    config_path = tmp_path / "agent-config.yml"
    config_path.write_text(
        """
rate_limit_policy:
  database: ~/proxy-state.sqlite3
  plans:
    claude-cli: pro
    google: free
"""
    )

    policy = load_rate_limit_policy(config_path)

    assert policy["database_path"].name == "proxy-state.sqlite3"
    assert policy["configured_plans"] == {
        "claude-cli": "pro",
        "google": "free",
    }


def test_store_caches_models_and_policy_metadata(tmp_path):
    store = RateLimitStore(tmp_path / "state.sqlite3")
    store.configure_plan("openrouter", "free")
    store.register_models(
        ["openrouter/model-a:free", "openrouter/model-b:free"]
    )

    rows = store.inventory()

    assert [row["model"] for row in rows] == ["model-a:free", "model-b:free"]
    assert rows[0]["plan"] == "free"
    assert rows[0]["default_cooldown_seconds"] == 60
    assert rows[0]["source_url"].startswith("https://openrouter.ai/")


@pytest.mark.anyio
async def test_plan_cooldown_is_persisted_and_filters_until_expiry(tmp_path):
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    callback = PersistentRateLimitCallback(
        database_path=tmp_path / "state.sqlite3",
        configured_plans={"claude-cli": "pro", "github-copilot": "unlimited"},
        model_keys=[
            "claude-cli/claude-sonnet-5",
            "github-copilot/gpt-5.5",
        ],
        clock=lambda: now,
    )
    claude = deployment("claude-cli/claude-sonnet-5")
    copilot = deployment("github-copilot/gpt-5.5")

    await callback.async_log_failure_event(
        {
            "exception": RateLimitedError(),
            "litellm_params": {
                "model_info": {"rate_limit_key": "claude-cli/claude-sonnet-5"}
            },
        },
        None,
        None,
        None,
    )

    with sqlite3.connect(callback.store.database_path) as connection:
        row = connection.execute(
            """
            SELECT last_rate_limited_at, retry_at, retry_source
            FROM rate_limits
            """
        ).fetchone()
    assert row == (
        "2026-07-10T00:00:00Z",
        "2026-07-10T05:00:00Z",
        "plan:pro",
    )
    assert await callback.async_filter_deployments(
        "chain", [claude, copilot], None
    ) == [copilot]

    callback._clock = lambda: now + timedelta(hours=5)
    assert await callback.async_filter_deployments(
        "chain", [claude, copilot], None
    ) == [claude, copilot]


@pytest.mark.anyio
async def test_provider_retry_after_overrides_plan_default(tmp_path):
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    callback = PersistentRateLimitCallback(
        database_path=tmp_path / "state.sqlite3",
        configured_plans={"claude-cli": "pro"},
        clock=lambda: now,
    )

    await callback.async_log_failure_event(
        {
            "exception": WrappedProviderError(
                RateLimitedError({"Retry-After": "90"})
            ),
            "litellm_params": {
                "model_info": {"rate_limit_key": "claude-cli/claude-sonnet-5"}
            },
        },
        None,
        None,
        None,
    )

    assert callback.store.retry_at(
        "claude-cli/claude-sonnet-5"
    ) == now + timedelta(seconds=90)


@pytest.mark.anyio
async def test_non_rate_limit_does_not_create_event(tmp_path):
    callback = PersistentRateLimitCallback(
        database_path=tmp_path / "state.sqlite3",
        configured_plans={"claude-cli": "pro"},
    )

    await callback.async_log_failure_event(
        {
            "exception": RuntimeError("provider failed"),
            "litellm_params": {
                "model_info": {"rate_limit_key": "claude-cli/claude-sonnet-5"}
            },
        },
        None,
        None,
        None,
    )

    with sqlite3.connect(callback.store.database_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM rate_limits").fetchone()[0]
    assert count == 0
