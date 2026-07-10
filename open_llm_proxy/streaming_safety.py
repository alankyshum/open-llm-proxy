from __future__ import annotations

from functools import wraps


def install_pre_first_chunk_fallback_only() -> None:
    """Prevent provider fallback after streamed output may have caused side effects."""
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    original = CustomStreamWrapper._handle_stream_fallback_error
    if getattr(original, "_open_llm_proxy_safe_fallback", False):
        return

    @wraps(original)
    def handle_stream_error(self, error):
        if self.sent_first_chunk:
            raise error
        return original(self, error)

    handle_stream_error._open_llm_proxy_safe_fallback = True
    CustomStreamWrapper._handle_stream_fallback_error = handle_stream_error
