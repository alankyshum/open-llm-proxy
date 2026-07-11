import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from litellm.exceptions import MidStreamFallbackError

from open_llm_proxy.config_gen import generate_config
from open_llm_proxy.rate_limit_state import (
    PersistentRateLimitCallback,
    RateLimitStore,
    load_rate_limit_policy,
)


class RateLimitedError(Exception):
    status_code = 429

    def __init__(self, headers=None, *, origin_key=None, provider=None):
        super().__init__("rate limited")
        self.headers = headers or {}
        self.rate_limit_origin_key = origin_key
        self.llm_provider = provider


class WrappedProviderError(Exception):
    status_code = 503

    def __init__(self, original_exception):
        super().__init__("provider stream failed")
        self.original_exception = original_exception


class ProviderRateLimitedError(RateLimitedError):
    def __init__(self, provider, headers=None):
        super().__init__(headers)
        self.llm_provider = provider


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
    store = RateLimitStore(tmp_path / "state.sqlite3", configured_plans={"openrouter": "free"})
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
            "exception": RateLimitedError(
                origin_key="claude-cli/claude-sonnet-5"
            ),
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
async def test_generated_chain_keeps_ordered_fallback_after_primary_cooldown(tmp_path):
    config_path = tmp_path / "agent-config.yml"
    config_path.write_text(
        """
file_settings:
  opencode:
    model: "open-llm-proxy/[google/gemini-3.5-flash,github-copilot/gemini-3.5-flash]"
"""
    )
    config = generate_config(str(config_path))
    alias = "[google/gemini-3.5-flash;github-copilot/gemini-3.5-flash]"
    chain = [d for d in config["model_list"] if d["model_name"] == alias]
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    callback = PersistentRateLimitCallback(
        database_path=tmp_path / "state.sqlite3",
        configured_plans={"google": "free", "github-copilot": "unlimited"},
        clock=lambda: now,
    )

    await callback.async_log_failure_event(
        {
            "exception": RateLimitedError(
                origin_key="google/gemini-3.5-flash"
            ),
            "litellm_params": {"model_info": chain[0]["model_info"]},
        },
        None,
        None,
        None,
    )

    available = await callback.async_filter_deployments(alias, chain, None)
    assert [d["model_info"]["rate_limit_key"] for d in available] == [
        "github-copilot/gemini-3.5-flash"
    ]
    assert available[0]["litellm_params"]["order"] == 2


@pytest.mark.anyio
async def test_wrapped_fallback_rate_limit_does_not_poison_current_model(tmp_path):
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

    assert callback.store.retry_at("claude-cli/claude-sonnet-5") is None


@pytest.mark.anyio
async def test_midstream_fallback_rate_limit_does_not_poison_current_model(tmp_path):
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    callback = PersistentRateLimitCallback(
        database_path=tmp_path / "state.sqlite3",
        configured_plans={"github-copilot": "unlimited"},
        clock=lambda: now,
    )
    error = MidStreamFallbackError(
        message="previous deployment was rate limited",
        model="gemini-3.5-flash",
        llm_provider="github-copilot",
        original_exception=RateLimitedError({"Retry-After": "60"}),
        is_pre_first_chunk=True,
    )

    await callback.async_log_failure_event(
        {
            "exception": error,
            "litellm_params": {
                "model_info": {
                    "rate_limit_key": "github-copilot/gemini-3.5-flash"
                }
            },
        },
        None,
        None,
        None,
    )

    assert callback.store.retry_at("github-copilot/gemini-3.5-flash") is None


@pytest.mark.anyio
async def test_google_rate_limit_does_not_poison_copilot_fallback_metadata(tmp_path):
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    callback = PersistentRateLimitCallback(
        database_path=tmp_path / "state.sqlite3",
        configured_plans={"github-copilot": "unlimited"},
        clock=lambda: now,
    )

    await callback.async_log_failure_event(
        {
            "exception": ProviderRateLimitedError("gemini"),
            "litellm_params": {
                "model_info": {
                    "rate_limit_key": "github-copilot/gemini-3.5-flash"
                }
            },
        },
        None,
        None,
        None,
    )

    assert callback.store.retry_at("github-copilot/gemini-3.5-flash") is None


@pytest.mark.anyio
async def test_custom_provider_origin_overrides_mutated_fallback_metadata(tmp_path):
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    callback = PersistentRateLimitCallback(
        database_path=tmp_path / "state.sqlite3",
        configured_plans={
            "google": "free",
            "github-copilot": "unlimited",
        },
        clock=lambda: now,
    )
    error = RateLimitedError()
    error.rate_limit_origin_key = "github-copilot/gemini-3.5-flash"

    await callback.async_log_failure_event(
        {
            "exception": error,
            "litellm_params": {
                "model_info": {"rate_limit_key": "google/gemini-3.5-flash"}
            },
        },
        None,
        None,
        None,
    )

    assert callback.store.retry_at("google/gemini-3.5-flash") is None
    assert callback.store.retry_at(
        "github-copilot/gemini-3.5-flash"
    ) == now + timedelta(seconds=60)


@pytest.mark.anyio
async def test_direct_provider_retry_after_overrides_plan_default(tmp_path):
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    callback = PersistentRateLimitCallback(
        database_path=tmp_path / "state.sqlite3",
        configured_plans={"claude-cli": "pro"},
        clock=lambda: now,
    )

    await callback.async_log_failure_event(
        {
            "exception": RateLimitedError(
                {"Retry-After": "90"},
                origin_key="claude-cli/claude-sonnet-5",
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
async def test_all_persistently_limited_deployments_raise_specific_error(tmp_path):
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    callback = PersistentRateLimitCallback(
        database_path=tmp_path / "state.sqlite3",
        configured_plans={"claude-cli": "pro"},
        clock=lambda: now,
    )
    key = "claude-cli/claude-fable-5"
    callback.store.record_rate_limit(
        key,
        occurred_at=now,
        retry_at=now + timedelta(minutes=90),
        retry_source="header:retry-after",
    )

    with pytest.raises(Exception) as exc_info:
        await callback.async_filter_deployments(
            "claude-fable", [deployment(key)], None
        )

    error = exc_info.value
    assert getattr(error, "status_code", None) == 429
    assert getattr(error, "proxy_persistent_rate_limit", False)
    assert "Proxy is running" in str(error)
    assert key in str(error)
    assert "2026-07-10T01:30:00Z" in str(error)
    assert "header:retry-after" in str(error)


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
