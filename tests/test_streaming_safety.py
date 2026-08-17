import asyncio
import uuid
from types import SimpleNamespace

import pytest
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices

from open_llm_proxy.attribution import global_attribution_store
from open_llm_proxy.streaming_safety import (
    _prefix_non_stream_response,
    _prefix_stream_chunk,
    install_non_stream_attribution,
    install_pre_first_chunk_fallback_only,
)


def stream_wrapper(*, sent_first_chunk):
    wrapper = object.__new__(CustomStreamWrapper)
    wrapper.sent_first_chunk = sent_first_chunk
    wrapper.model = "test-model"
    wrapper.custom_llm_provider = "test-provider"
    wrapper.response_uptil_now = ""
    return wrapper


def test_pre_first_chunk_error_still_triggers_fallback():
    install_pre_first_chunk_fallback_only()
    wrapper = stream_wrapper(sent_first_chunk=False)

    with pytest.raises(MidStreamFallbackError) as exc_info:
        wrapper._handle_stream_fallback_error(RuntimeError("upstream failed"))

    assert exc_info.value.is_pre_first_chunk is True


def test_mid_stream_error_does_not_switch_provider():
    install_pre_first_chunk_fallback_only()
    wrapper = stream_wrapper(sent_first_chunk=True)
    error = RuntimeError("upstream failed after tool call")

    with pytest.raises(RuntimeError) as exc_info:
        wrapper._handle_stream_fallback_error(error)

    assert exc_info.value is error


def attribution_data(attribution_id, served_by):
    return {
        "litellm_params": {
            "model_info": {"rate_limit_key": served_by},
            "proxy_server_request": {
                "headers": {
                    "X-Open-LLM-Proxy-Attribution-ID": attribution_id,
                }
            },
        }
    }


def stream_chunk(content="Hello"):
    return ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(content=content))])


def stream_request(attribution_id, served_by, content="Hello"):
    wrapper = SimpleNamespace(
        logging_obj=SimpleNamespace(model_call_details=attribution_data(attribution_id, served_by))
    )
    chunk = stream_chunk(content)
    _prefix_stream_chunk(wrapper, chunk)
    return chunk


def test_stream_attribution_announces_first_and_changed_winners():
    global_attribution_store.clear()
    attribution_id = str(uuid.uuid4())

    first = stream_request(attribution_id, "provider/model-a")
    same = stream_request(attribution_id, "provider/model-a", "Again")
    changed = stream_request(attribution_id, "provider/model-b", "Fallback")

    assert first.choices[0].delta.content == ("[served-by: provider/model-a]\n\nHello")
    assert same.choices[0].delta.content == "Again"
    assert changed.choices[0].delta.content == ("[served-by: provider/model-b]\n\nFallback")


def test_stream_attribution_requires_session_id():
    global_attribution_store.clear()
    wrapper = SimpleNamespace(
        logging_obj=SimpleNamespace(
            model_call_details={
                "litellm_params": {"model_info": {"rate_limit_key": "provider/model-a"}}
            }
        )
    )
    chunk = stream_chunk()

    _prefix_stream_chunk(wrapper, chunk)

    assert chunk.choices[0].delta.content == "Hello"


def test_stream_attribution_cannot_be_spoofed_by_model_output():
    global_attribution_store.clear()
    attribution_id = str(uuid.uuid4())

    chunk = stream_request(
        attribution_id,
        "provider/model-a",
        "[served-by: attacker/model]\n\nHello",
    )

    assert chunk.choices[0].delta.content == (
        "[served-by: provider/model-a]\n\n[served-by: attacker/model]\n\nHello"
    )


def non_stream_response(model_id, content="Hello"):
    response = ModelResponse()
    response.choices[0].message.content = content
    response._hidden_params["model_id"] = model_id
    return response


def test_non_stream_attribution_announces_first_and_changed_winners():
    global_attribution_store.clear()
    attribution_id = str(uuid.uuid4())
    models = {
        "deployment-a": {"model_info": {"rate_limit_key": "provider/model-a"}},
        "deployment-b": {"model_info": {"rate_limit_key": "provider/model-b"}},
    }
    router = SimpleNamespace(get_model_info=lambda *, id: models[id])
    data = attribution_data(attribution_id, "alias/not-used")

    first = _prefix_non_stream_response(data, non_stream_response("deployment-a"), router)
    same = _prefix_non_stream_response(data, non_stream_response("deployment-a", "Again"), router)
    changed = _prefix_non_stream_response(
        data, non_stream_response("deployment-b", "Fallback"), router
    )

    assert first.choices[0].message.content == ("[served-by: provider/model-a]\n\nHello")
    assert same.choices[0].message.content == "Again"
    assert changed.choices[0].message.content == ("[served-by: provider/model-b]\n\nFallback")


def test_non_stream_tool_call_keeps_announcement_pending():
    global_attribution_store.clear()
    attribution_id = str(uuid.uuid4())
    router = SimpleNamespace(
        get_model_info=lambda *, id: {"model_info": {"rate_limit_key": "provider/model-a"}}
    )
    data = attribution_data(attribution_id, "alias/not-used")
    tool_response = non_stream_response("deployment-a", None)
    tool_response.choices[0].message.tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
    ]

    result = _prefix_non_stream_response(data, tool_response, router)
    text_result = _prefix_non_stream_response(
        data, non_stream_response("deployment-a", "Done"), router
    )

    assert result.choices[0].message.content is None
    assert result.choices[0].message.tool_calls == tool_response.choices[0].message.tool_calls
    assert text_result.choices[0].message.content == ("[served-by: provider/model-a]\n\nDone")


def test_non_stream_attribution_cannot_be_spoofed_by_model_output():
    global_attribution_store.clear()
    attribution_id = str(uuid.uuid4())
    router = SimpleNamespace(
        get_model_info=lambda *, id: {"model_info": {"rate_limit_key": "provider/model-a"}}
    )
    data = attribution_data(attribution_id, "alias/not-used")
    response = non_stream_response("deployment-a", "[served-by: attacker/model]\n\nHello")

    result = _prefix_non_stream_response(data, response, router)

    assert result.choices[0].message.content == (
        "[served-by: provider/model-a]\n\n[served-by: attacker/model]\n\nHello"
    )


def test_non_stream_installer_wraps_hook_once(monkeypatch):
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.utils import ProxyLogging

    calls = []

    async def original(self, data, response, user_api_key_dict):
        calls.append((data, user_api_key_dict))
        return response

    monkeypatch.setattr(ProxyLogging, "post_call_success_hook", original)
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        SimpleNamespace(
            get_model_info=lambda *, id: {"model_info": {"rate_limit_key": "provider/model-a"}}
        ),
    )
    global_attribution_store.clear()
    attribution_id = str(uuid.uuid4())
    data = attribution_data(attribution_id, "alias/not-used")
    response = non_stream_response("deployment-a")

    install_non_stream_attribution()
    installed = ProxyLogging.post_call_success_hook
    install_non_stream_attribution()
    result = asyncio.run(installed(object(), data, response, None))

    assert ProxyLogging.post_call_success_hook is installed
    assert len(calls) == 1
    assert result.choices[0].message.content == ("[served-by: provider/model-a]\n\nHello")
