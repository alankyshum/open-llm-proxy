from __future__ import annotations

from functools import wraps

from open_llm_proxy.callbacks import (
    is_first_assistant_turn,
    prefix_response_content,
    served_by_banner,
    served_by_response_banner,
)


def _prefix_stream_chunk(wrapper, response):
    if response is None or getattr(wrapper, "_open_llm_proxy_attributed", False):
        return response

    details = getattr(getattr(wrapper, "logging_obj", None), "model_call_details", None)
    if isinstance(details, dict) and not is_first_assistant_turn(details):
        wrapper._open_llm_proxy_attributed = True
        return response
    banner = served_by_banner(details) if isinstance(details, dict) else None
    if banner is None:
        return response

    choices = getattr(response, "choices", None)
    if not choices:
        return response
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return response

    content = getattr(delta, "content", None)
    if isinstance(content, str) and content.startswith("[served-by: "):
        wrapper._open_llm_proxy_attributed = True
        return response

    # Content may be None on role-only and tool-call chunks. OpenAI-compatible
    # deltas permit text alongside those fields, so attribution can still lead.
    delta.content = banner + (content or "")
    wrapper._open_llm_proxy_attributed = True
    return response


def install_non_stream_attribution() -> None:
    """Prefix completed responses at LiteLLM's final mutable response boundary."""
    from litellm.proxy.utils import ProxyLogging

    original = ProxyLogging.post_call_success_hook
    if getattr(original, "_open_llm_proxy_served_by", False):
        return

    @wraps(original)
    async def post_call_success_hook(self, data, response, user_api_key_dict):
        result = await original(self, data, response, user_api_key_dict)
        if not is_first_assistant_turn(data):
            return result
        from litellm.proxy.proxy_server import llm_router

        banner = served_by_response_banner(result, llm_router)
        return prefix_response_content(result, banner) if banner else result

    post_call_success_hook._open_llm_proxy_served_by = True
    ProxyLogging.post_call_success_hook = post_call_success_hook


def install_pre_first_chunk_fallback_only() -> None:
    """Install fallback safety and earliest winning-deployment attribution."""
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    original = CustomStreamWrapper._handle_stream_fallback_error
    if not getattr(original, "_open_llm_proxy_safe_fallback", False):
        @wraps(original)
        def handle_stream_error(self, error):
            if self.sent_first_chunk:
                raise error
            return original(self, error)

        handle_stream_error._open_llm_proxy_safe_fallback = True
        CustomStreamWrapper._handle_stream_fallback_error = handle_stream_error

    original_chunk_creator = CustomStreamWrapper.chunk_creator
    if not getattr(original_chunk_creator, "_open_llm_proxy_served_by", False):
        @wraps(original_chunk_creator)
        def chunk_creator(self, chunk):
            return _prefix_stream_chunk(self, original_chunk_creator(self, chunk))

        chunk_creator._open_llm_proxy_served_by = True
        CustomStreamWrapper.chunk_creator = chunk_creator
