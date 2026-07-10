import pytest
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

from open_llm_proxy.streaming_safety import install_pre_first_chunk_fallback_only


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
