import asyncio
from types import SimpleNamespace
from typing import Any
import pytest

from open_llm_proxy.callbacks import ServedByCallback

# Use a mock response object resembling LiteLLM's ModelResponse
class MockResponse:
    def __init__(self, content: str = "Hello", model: str = "open-llm-proxy/alias", tool_calls: Any = None):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls
                )
            )
        ]
        self.model = model
        self._hidden_params = {}


def test_selected_deployment_exact_header():
    callback = ServedByCallback()
    data = {
        "deployment": {
            "model_info": {
                "rate_limit_key": "openai/gpt-4"
            }
        }
    }
    headers = asyncio.run(callback.async_post_call_response_headers_hook(data, None, MockResponse()))
    assert headers == {"x-open-llm-proxy-served-by": "openai/gpt-4"}


def test_hidden_model_id_fallback(monkeypatch):
    callback = ServedByCallback()
    data = {}
    response = MockResponse()
    response._hidden_params = {"model_id": "test-deployment-id"}

    # Mock the router
    mock_router = SimpleNamespace(
        get_model_info=lambda *, id: {
            "model_info": {
                "rate_limit_key": "anthropic/claude-3.5-sonnet"
            }
        } if id == "test-deployment-id" else None
    )

    import litellm.proxy.proxy_server as proxy_server
    monkeypatch.setattr(proxy_server, "llm_router", mock_router)

    headers = asyncio.run(callback.async_post_call_response_headers_hook(data, None, response))
    assert headers == {"x-open-llm-proxy-served-by": "anthropic/claude-3.5-sonnet"}


def test_no_metadata_empty_mapping():
    callback = ServedByCallback()
    # Empty data and no hidden params
    headers = asyncio.run(callback.async_post_call_response_headers_hook({}, None, MockResponse()))
    assert headers == {}


def test_streaming_and_nonstream_payload_unchanged():
    callback = ServedByCallback()
    response = MockResponse(content="Keep original content untouched")
    data = {
        "deployment": {
            "model_info": {
                "rate_limit_key": "google/gemini-1.5"
            }
        }
    }
    _ = asyncio.run(callback.async_post_call_response_headers_hook(data, None, response))
    
    # Assert that response properties remain unchanged
    assert response.choices[0].message.content == "Keep original content untouched"


def test_tool_calls_unchanged():
    callback = ServedByCallback()
    tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "get_weather"}}]
    response = MockResponse(content=None, tool_calls=tool_calls)
    data = {
        "deployment": {
            "model_info": {
                "rate_limit_key": "google/gemini-1.5"
            }
        }
    }
    _ = asyncio.run(callback.async_post_call_response_headers_hook(data, None, response))
    
    # Assert tool_calls are unchanged
    assert response.choices[0].message.tool_calls == tool_calls


def test_response_model_remains_alias():
    callback = ServedByCallback()
    response = MockResponse(model="my-cool-alias")
    data = {
        "deployment": {
            "model_info": {
                "rate_limit_key": "openai/gpt-4"
            }
        }
    }
    _ = asyncio.run(callback.async_post_call_response_headers_hook(data, None, response))
    
    assert response.model == "my-cool-alias"


# ── Regression: NO body-mutation hooks or banner content ──────────────

def test_no_post_call_success_hook():
    """ServedByCallback MUST NOT override the body-mutating async_post_call_success_hook."""
    assert "async_post_call_success_hook" not in ServedByCallback.__dict__, (
        "Body-mutating hook must not be overridden in ServedByCallback"
    )


def test_no_served_by_prefix_constant():
    """The _SERVED_BY_PREFIX constant must not exist in callbacks module."""
    import open_llm_proxy.callbacks as mod
    assert not hasattr(mod, "_SERVED_BY_PREFIX"), (
        "Banner prefix constant must be removed"
    )


def test_no_served_by_content_in_body():
    """Header hook must NOT leave [served-by: in response body content."""
    callback = ServedByCallback()
    original = "This is a normal assistant reply without attribution."
    response = MockResponse(content=original)
    data = {
        "deployment": {
            "model_info": {
                "rate_limit_key": "openai/gpt-4"
            }
        }
    }
    _ = asyncio.run(callback.async_post_call_response_headers_hook(
        data, None, response
    ))
    content = response.choices[0].message.content
    assert "[served-by:" not in content, (
        f"Response body must not contain banner text; got: {content!r}"
    )
    assert content == original, "Content must be byte-identical"


def test_no_banner_for_streaming_and_tool_calls():
    """Verify no banner leaks with streaming-like or tool-call responses."""
    callback = ServedByCallback()
    # Tool-call response (content=None, tool_calls present)
    tool_calls = [{"id": "tc1", "type": "function", "function": {"name": "f"}}]
    response = MockResponse(content=None, tool_calls=tool_calls)
    data = {
        "deployment": {
            "model_info": {
                "rate_limit_key": "openai/gpt-4"
            }
        }
    }
    _ = asyncio.run(callback.async_post_call_response_headers_hook(
        data, None, response
    ))
    # Content is None, tool_calls must be unchanged
    assert response.choices[0].message.content is None
    assert response.choices[0].message.tool_calls == tool_calls


def test_no_body_mutation_banner_helpers_gone():
    """Ensure banner helper functions were removed from the module."""
    import open_llm_proxy.callbacks as mod
    for name in ("served_by_banner", "served_by_response_banner",
                 "prefix_response_content"):
        assert not hasattr(mod, name), (
            f"Dead banner helper {name} must be removed"
            )
