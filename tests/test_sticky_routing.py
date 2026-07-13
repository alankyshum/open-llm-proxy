import os
import uuid
import asyncio
import pytest
from open_llm_proxy.attribution import global_attribution_store
from open_llm_proxy.callbacks import StickyRoutingCallback


def test_pin_when_present():
    attr_id = str(uuid.uuid4())
    winner = "github-copilot/gemini-3.5-flash"
    global_attribution_store.set(attr_id, winner)

    healthy_deployments = [
        {"model_name": "other-1", "model_info": {"rate_limit_key": "openai/gpt-4"}},
        {"model_name": "gemini", "model_info": {"rate_limit_key": winner}},
        {"model_name": "other-2", "model_info": {"rate_limit_key": "anthropic/claude-3-5"}},
    ]

    request_kwargs = {
        "proxy_server_request": {
            "headers": {
                "x-open-llm-proxy-attribution-id": attr_id
            }
        }
    }

    callback = StickyRoutingCallback()
    res = asyncio.run(callback.async_filter_deployments(
        model="some-model",
        healthy_deployments=healthy_deployments,
        messages=[],
        request_kwargs=request_kwargs
    ))

    assert len(res) == 1
    assert res[0]["model_info"]["rate_limit_key"] == winner


def test_passthrough_when_winner_absent():
    attr_id = str(uuid.uuid4())
    winner = "github-copilot/gemini-3.5-flash"
    global_attribution_store.set(attr_id, winner)

    healthy_deployments = [
        {"model_name": "other-1", "model_info": {"rate_limit_key": "openai/gpt-4"}},
        {"model_name": "other-2", "model_info": {"rate_limit_key": "anthropic/claude-3-5"}},
    ]

    request_kwargs = {
        "proxy_server_request": {
            "headers": {
                "x-open-llm-proxy-attribution-id": attr_id
            }
        }
    }

    callback = StickyRoutingCallback()
    res = asyncio.run(callback.async_filter_deployments(
        model="some-model",
        healthy_deployments=healthy_deployments,
        messages=[],
        request_kwargs=request_kwargs
    ))

    assert res == healthy_deployments


def test_first_turn_passthrough():
    attr_id = str(uuid.uuid4())
    # no seeding in global_attribution_store

    healthy_deployments = [
        {"model_name": "other-1", "model_info": {"rate_limit_key": "openai/gpt-4"}},
        {"model_name": "gemini", "model_info": {"rate_limit_key": "github-copilot/gemini-3.5-flash"}},
    ]

    request_kwargs = {
        "proxy_server_request": {
            "headers": {
                "x-open-llm-proxy-attribution-id": attr_id
            }
        }
    }

    callback = StickyRoutingCallback()
    res = asyncio.run(callback.async_filter_deployments(
        model="some-model",
        healthy_deployments=healthy_deployments,
        messages=[],
        request_kwargs=request_kwargs
    ))

    assert res == healthy_deployments


def test_disabled_via_env(monkeypatch):
    monkeypatch.setenv("OPEN_LLM_PROXY_STICKY_ROUTING", "0")
    attr_id = str(uuid.uuid4())
    winner = "github-copilot/gemini-3.5-flash"
    global_attribution_store.set(attr_id, winner)

    healthy_deployments = [
        {"model_name": "other-1", "model_info": {"rate_limit_key": "openai/gpt-4"}},
        {"model_name": "gemini", "model_info": {"rate_limit_key": winner}},
    ]

    request_kwargs = {
        "proxy_server_request": {
            "headers": {
                "x-open-llm-proxy-attribution-id": attr_id
            }
        }
    }

    callback = StickyRoutingCallback()
    res = asyncio.run(callback.async_filter_deployments(
        model="some-model",
        healthy_deployments=healthy_deployments,
        messages=[],
        request_kwargs=request_kwargs
    ))

    assert res == healthy_deployments


def test_single_or_empty_deployment():
    attr_id = str(uuid.uuid4())
    winner = "github-copilot/gemini-3.5-flash"
    global_attribution_store.set(attr_id, winner)

    request_kwargs = {
        "proxy_server_request": {
            "headers": {
                "x-open-llm-proxy-attribution-id": attr_id
            }
        }
    }

    callback = StickyRoutingCallback()

    # Empty list
    res_empty = asyncio.run(callback.async_filter_deployments(
        model="some-model",
        healthy_deployments=[],
        messages=[],
        request_kwargs=request_kwargs
    ))
    assert res_empty == []

    # Single deployment
    single = [{"model_name": "gemini", "model_info": {"rate_limit_key": winner}}]
    res_single = asyncio.run(callback.async_filter_deployments(
        model="some-model",
        healthy_deployments=single,
        messages=[],
        request_kwargs=request_kwargs
    ))
    assert res_single == single
