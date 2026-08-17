import asyncio
from types import SimpleNamespace
from typing import Any

from open_llm_proxy.callbacks import ServedByCallback


class MockResponse:
    def __init__(
        self,
        content: str = "Hello",
        model: str = "open-llm-proxy/alias",
        tool_calls: Any = None,
    ):
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))
        ]
        self.model = model
        self._hidden_params = {}


def test_selected_deployment_exact_header():
    callback = ServedByCallback()
    data = {"deployment": {"model_info": {"rate_limit_key": "openai/gpt-4"}}}
    headers = asyncio.run(
        callback.async_post_call_response_headers_hook(data, None, MockResponse())
    )
    assert headers == {"x-open-llm-proxy-served-by": "openai/gpt-4"}


def test_hidden_model_id_fallback(monkeypatch):
    callback = ServedByCallback()
    response = MockResponse()
    response._hidden_params = {"model_id": "test-deployment-id"}
    mock_router = SimpleNamespace(
        get_model_info=lambda *, id: (
            {"model_info": {"rate_limit_key": "anthropic/claude-3.5-sonnet"}}
            if id == "test-deployment-id"
            else None
        )
    )

    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", mock_router)
    headers = asyncio.run(callback.async_post_call_response_headers_hook({}, None, response))
    assert headers == {"x-open-llm-proxy-served-by": "anthropic/claude-3.5-sonnet"}


def test_no_metadata_empty_mapping():
    callback = ServedByCallback()
    headers = asyncio.run(callback.async_post_call_response_headers_hook({}, None, MockResponse()))
    assert headers == {}


def test_streaming_and_nonstream_payload_unchanged():
    callback = ServedByCallback()
    response = MockResponse(content="Keep original content untouched")
    data = {"deployment": {"model_info": {"rate_limit_key": "google/gemini-1.5"}}}
    asyncio.run(callback.async_post_call_response_headers_hook(data, None, response))
    assert response.choices[0].message.content == "Keep original content untouched"


def test_tool_calls_unchanged():
    callback = ServedByCallback()
    tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "get_weather"}}]
    response = MockResponse(content=None, tool_calls=tool_calls)
    data = {"deployment": {"model_info": {"rate_limit_key": "google/gemini-1.5"}}}
    asyncio.run(callback.async_post_call_response_headers_hook(data, None, response))
    assert response.choices[0].message.tool_calls == tool_calls


def test_response_model_remains_alias():
    callback = ServedByCallback()
    response = MockResponse(model="my-cool-alias")
    data = {"deployment": {"model_info": {"rate_limit_key": "openai/gpt-4"}}}
    asyncio.run(callback.async_post_call_response_headers_hook(data, None, response))
    assert response.model == "my-cool-alias"


def test_no_post_call_success_hook():
    assert "async_post_call_success_hook" not in ServedByCallback.__dict__


def test_no_served_by_prefix_constant():
    import open_llm_proxy.callbacks as callbacks

    assert not hasattr(callbacks, "_SERVED_BY_PREFIX")


def test_no_served_by_content_in_body():
    callback = ServedByCallback()
    original = "This is a normal assistant reply without attribution."
    response = MockResponse(content=original)
    data = {"deployment": {"model_info": {"rate_limit_key": "openai/gpt-4"}}}
    asyncio.run(callback.async_post_call_response_headers_hook(data, None, response))
    assert "[served-by:" not in response.choices[0].message.content
    assert response.choices[0].message.content == original


def test_no_banner_for_streaming_and_tool_calls():
    callback = ServedByCallback()
    tool_calls = [{"id": "tc1", "type": "function", "function": {"name": "f"}}]
    response = MockResponse(content=None, tool_calls=tool_calls)
    data = {"deployment": {"model_info": {"rate_limit_key": "openai/gpt-4"}}}
    asyncio.run(callback.async_post_call_response_headers_hook(data, None, response))
    assert response.choices[0].message.content is None
    assert response.choices[0].message.tool_calls == tool_calls


def test_no_body_mutation_banner_helpers_gone():
    import open_llm_proxy.callbacks as callbacks

    for name in (
        "served_by_banner",
        "served_by_response_banner",
        "prefix_response_content",
    ):
        assert not hasattr(callbacks, name)
