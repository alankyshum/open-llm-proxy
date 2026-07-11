import asyncio
from types import SimpleNamespace

from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices

from open_llm_proxy.callbacks import (
    ServedByCallback,
    is_first_assistant_turn,
    served_by_banner,
    served_by_response_banner,
)
from open_llm_proxy.streaming_safety import (
    _prefix_stream_chunk,
    install_non_stream_attribution,
)


def request_data(key="google/gemini-3.5-flash", messages=None):
    return {
        "messages": messages or [{"role": "user", "content": "Hello"}],
        "litellm_params": {"model_info": {"rate_limit_key": key}},
    }


def test_first_assistant_turn_uses_request_history():
    assert is_first_assistant_turn(request_data())
    assert not is_first_assistant_turn(request_data(messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Again"},
    ]))


def test_served_by_banner_uses_concrete_deployment_key():
    assert served_by_banner(request_data()) == (
        "[served-by: google/gemini-3.5-flash]\n\n"
    )
    assert served_by_banner({"model_info": {"rate_limit_key": "openai/gpt-5"}}) == (
        "[served-by: openai/gpt-5]\n\n"
    )
    assert served_by_banner({}) is None


def test_non_streaming_response_is_prefixed():
    response = ModelResponse()
    response.choices[0].message.content = "Hello"

    result = asyncio.run(
        ServedByCallback().async_post_call_success_hook(request_data(), None, response)
    )

    assert result.choices[0].message.content == (
        "[served-by: google/gemini-3.5-flash]\n\nHello"
    )


def test_non_streaming_tool_call_response_gets_text_without_losing_tools():
    response = ModelResponse()
    response.choices[0].message.content = None
    response.choices[0].message.tool_calls = [{
        "id": "call-1",
        "type": "function",
        "function": {"name": "status", "arguments": "{}"},
    }]

    result = asyncio.run(
        ServedByCallback().async_post_call_success_hook(request_data(), None, response)
    )

    assert result.choices[0].message.content.startswith("[served-by: ")
    assert result.choices[0].message.tool_calls[0]["function"]["name"] == "status"


def test_non_streaming_later_turn_is_not_prefixed():
    response = ModelResponse()
    response.choices[0].message.content = "Again"
    data = request_data(messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Again"},
    ])

    result = asyncio.run(
        ServedByCallback().async_post_call_success_hook(data, None, response)
    )

    assert result.choices[0].message.content == "Again"


def test_response_banner_resolves_selected_deployment_id():
    response = ModelResponse()
    response._hidden_params["model_id"] = "deployment-1"
    router = SimpleNamespace(
        get_model_info=lambda *, id: {
            "model_info": {"rate_limit_key": "github-copilot/gpt-5.6-sol"}
        }
        if id == "deployment-1"
        else None
    )

    assert served_by_response_banner(response, router) == (
        "[served-by: github-copilot/gpt-5.6-sol]\n\n"
    )


def test_streaming_attribution_is_first_and_only_once():
    wrapper = SimpleNamespace(
        logging_obj=SimpleNamespace(model_call_details=request_data())
    )
    first = ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(role="assistant"))]
    )
    second = ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content="Hello"))]
    )

    _prefix_stream_chunk(wrapper, first)
    _prefix_stream_chunk(wrapper, second)

    assert first.choices[0].delta.content == (
        "[served-by: google/gemini-3.5-flash]\n\n"
    )
    assert second.choices[0].delta.content == "Hello"


def test_streaming_tool_call_kept_intact():
    wrapper = SimpleNamespace(
        logging_obj=SimpleNamespace(model_call_details=request_data())
    )
    chunk = ModelResponseStream(
        choices=[StreamingChoices(
            index=0,
            delta=Delta(tool_calls=[{
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"name": "status", "arguments": ""},
            }]),
        )]
    )

    _prefix_stream_chunk(wrapper, chunk)

    assert chunk.choices[0].delta.content.startswith("[served-by: ")
    assert chunk.choices[0].delta.tool_calls[0].function.name == "status"


def test_streaming_later_turn_is_not_prefixed():
    wrapper = SimpleNamespace(logging_obj=SimpleNamespace(model_call_details=request_data(
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Again"},
        ]
    )))
    chunk = ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content="Again"))]
    )

    _prefix_stream_chunk(wrapper, chunk)

    assert chunk.choices[0].delta.content == "Again"


def test_non_stream_hook_prefixes_proxy_response(monkeypatch):
    from litellm.proxy.utils import ProxyLogging
    from litellm.proxy import proxy_server

    original = ProxyLogging.post_call_success_hook

    async def base_hook(self, data, response, user_api_key_dict):
        return response

    monkeypatch.setattr(ProxyLogging, "post_call_success_hook", base_hook)
    install_non_stream_attribution()
    response = ModelResponse()
    response.choices[0].message.content = "Hello"
    response._hidden_params["model_id"] = "deployment-1"
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        SimpleNamespace(
            get_model_info=lambda *, id: {
                "model_info": {"rate_limit_key": "google/gemini-3.5-flash"}
            }
        ),
    )

    result = asyncio.run(
        ProxyLogging.post_call_success_hook(None, {}, response, None)
    )

    assert result.choices[0].message.content == (
        "[served-by: google/gemini-3.5-flash]\n\nHello"
    )
    monkeypatch.setattr(ProxyLogging, "post_call_success_hook", original)


def test_non_stream_hook_skips_later_turn(monkeypatch):
    from litellm.proxy.utils import ProxyLogging

    original = ProxyLogging.post_call_success_hook

    async def base_hook(self, data, response, user_api_key_dict):
        return response

    monkeypatch.setattr(ProxyLogging, "post_call_success_hook", base_hook)
    install_non_stream_attribution()
    response = ModelResponse()
    response.choices[0].message.content = "Again"
    data = request_data(messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Again"},
    ])

    result = asyncio.run(
        ProxyLogging.post_call_success_hook(None, data, response, None)
    )

    assert result.choices[0].message.content == "Again"
    monkeypatch.setattr(ProxyLogging, "post_call_success_hook", original)
