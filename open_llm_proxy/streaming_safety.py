from __future__ import annotations

from functools import wraps
import logging
from typing import Any

from open_llm_proxy.attribution import (
    attribution_id_from_data,
    global_attribution_store,
    served_by_from_data,
)


_SERVED_BY_PREFIX = "[served-by: "
log = logging.getLogger("open_llm_proxy.streaming_safety")


def _banner(served_by: str) -> str:
    return f"{_SERVED_BY_PREFIX}{served_by}]\n\n"


def _served_by_from_response(response: Any, router: Any) -> str | None:
    hidden_params = getattr(response, "_hidden_params", None) or {}
    model_id = hidden_params.get("model_id") if isinstance(hidden_params, dict) else None
    if not isinstance(model_id, str):
        return None
    try:
        deployment = router.get_model_info(id=model_id)
    except Exception:
        return None
    return served_by_from_data(deployment)


def _prefix_stream_chunk(wrapper: Any, response: Any) -> Any:
    if response is None or getattr(wrapper, "_open_llm_proxy_attribution_checked", False):
        return response

    choices = getattr(response, "choices", None)
    if not choices:
        return response
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return response

    details = getattr(getattr(wrapper, "logging_obj", None), "model_call_details", None)
    attr_id = attribution_id_from_data(details)
    served_by = served_by_from_data(details)
    wrapper._open_llm_proxy_attribution_checked = True
    if not attr_id or not served_by:
        return response

    content = getattr(delta, "content", None)
    if not global_attribution_store.announce_if_changed(attr_id, served_by):
        return response

    # Role-only and tool-call deltas may have no text. OpenAI-compatible
    # deltas permit text alongside those fields, so the banner can still lead.
    delta.content = _banner(served_by) + (content or "")
    return response


def _prefix_non_stream_response(data: dict, response: Any, router: Any) -> Any:
    attr_id = attribution_id_from_data(data)
    if not attr_id:
        return response

    choices = getattr(response, "choices", None)
    message = getattr(choices[0], "message", None) if choices else None
    if message is None:
        return response

    served_by = _served_by_from_response(response, router) or served_by_from_data(data)
    if not served_by:
        return response
    content = getattr(message, "content", None)
    if content is None:
        # Do not turn a tool-only response into a mixed text/tool response.
        # Preserve the unannounced winner so the next text response can show it.
        global_attribution_store.set(attr_id, served_by)
        return response
    if not global_attribution_store.announce_if_changed(attr_id, served_by):
        return response

    message.content = _banner(served_by) + (content or "")
    return response


def install_non_stream_attribution() -> None:
    """Prefix changed winners at LiteLLM's final mutable response boundary."""
    from litellm.proxy.utils import ProxyLogging

    original = ProxyLogging.post_call_success_hook
    if getattr(original, "_open_llm_proxy_served_by", False):
        return

    @wraps(original)
    async def post_call_success_hook(self, data, response, user_api_key_dict):
        result = await original(self, data, response, user_api_key_dict)
        try:
            from litellm.proxy.proxy_server import llm_router

            return _prefix_non_stream_response(data, result, llm_router)
        except Exception:
            log.exception("Failed to add non-stream model attribution")
            return result

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
            response = original_chunk_creator(self, chunk)
            try:
                return _prefix_stream_chunk(self, response)
            except Exception:
                log.exception("Failed to add streaming model attribution")
                return response

        chunk_creator._open_llm_proxy_served_by = True
        CustomStreamWrapper.chunk_creator = chunk_creator
