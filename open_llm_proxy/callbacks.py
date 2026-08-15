from typing import Any, Dict, List, Optional
import os
import re
import logging
import copy
from litellm.integrations.custom_logger import CustomLogger

from open_llm_proxy.content_parts import normalize_messages

from open_llm_proxy.attribution import (
    attribution_id_from_data,
    attribution_id_from_headers,
    global_attribution_store,
    served_by_from_data,
)

log = logging.getLogger("open_llm_proxy.callbacks")


class AttachmentContentNormalizationCallback(CustomLogger):
    """Ensure upstreams see only OpenAI ``text`` and ``image_url`` parts."""

    async def async_pre_call_hook(
        self, user_api_key_dict: Any, cache: Any, data: dict, call_type: str
    ) -> Optional[dict]:
        enabled = os.environ.get("OPEN_LLM_PROXY_NORMALIZE_ATTACHMENTS", "1")
        if enabled.lower() in ("0", "false", "no"):
            return None
        messages = data.get("messages")
        if not isinstance(messages, list):
            return None
        normalized = normalize_messages(messages)
        if normalized is messages:
            return None
        data["messages"] = normalized
        log.info("AttachmentContentNormalizationCallback: normalized attachment parts")
        return data

# Regex for extracting agent types from task description
_AGENT_LIST_RE = re.compile(
    r"Available agent types[^\n]*:\s*\n((?:[ \t]*-[ \t]*[A-Za-z][\w\-]*[^\n]*\n?)+)",
)
_AGENT_NAME_RE = re.compile(r"^[ \t]*-[ \t]*([A-Za-z][\w\-]*)\s*:", re.MULTILINE)

def _extract_agent_types(description: str) -> list[str]:
    if not isinstance(description, str) or not description:
        return []
    block_match = _AGENT_LIST_RE.search(description)
    if not block_match:
        return []
    block = block_match.group(1)
    names: list[str] = []
    seen: set[str] = set()
    for m in _AGENT_NAME_RE.finditer(block):
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names

def _rewrite_task_tool(tool: dict[str, Any]) -> bool:
    fn = tool.get("function")
    if not isinstance(fn, dict):
        return False
    if fn.get("name") != "task":
        return False
    params = fn.get("parameters")
    if not isinstance(params, dict):
        return False
    props = params.get("properties")
    if not isinstance(props, dict):
        return False
    sub = props.get("subagent_type")
    if not isinstance(sub, dict):
        return False
    if isinstance(sub.get("enum"), list) and sub["enum"]:
        return False

    names = _extract_agent_types(fn.get("description") or "")
    if not names:
        return False

    sub["enum"] = names
    base_desc = sub.get("description") or "The type of specialized agent to use for this task"
    sub["description"] = f"{base_desc}. Must be one of: {', '.join(names)}."
    return True


class GeminiThinkingBudgetCallback(CustomLogger):
    """
    Callback that floors max_tokens for models whose hidden reasoning tokens
    can consume the entire token budget, producing empty completions.
    """
    def __init__(self) -> None:
        super().__init__()
        # Each rule: (model_substring, default_floor, env_override_key_or_None)
        self._rules: list[tuple[str, int, str | None]] = [
            ("gemini",              1024, "OPEN_LLM_PROXY_GOOGLE_MIN_MAX_TOKENS"),
            ("google/",             1024, "OPEN_LLM_PROXY_GOOGLE_MIN_MAX_TOKENS"),
            ("deepseek-v4-flash-free",  4096, None),
            ("nemotron-3-ultra-free",   4096, None),
            ("nemotron-3-ultra-550b-a55b", 4096, None),
            ("gpt-5-mini",              4096, None),
            ("gpt-5.5",                 4096, None),
            ("glm-5.2",                 4096, None),
            ("kimi-k2.7-code",          4096, None),
            ("qwen3.7-plus",            4096, None),
            ("big-pickle",              4096, None),
        ]

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[dict]:
        model = data.get("model", "")
        model_lower = model.lower() if isinstance(model, str) else ""
        max_tokens = data.get("max_tokens")

        for pattern, default_floor, env_key in self._rules:
            if pattern not in model_lower:
                continue
            raw_floor = os.environ.get(env_key) if env_key is not None else None
            floor = default_floor
            if raw_floor is not None:
                try:
                    floor = max(0, min(int(raw_floor), 128_000))
                except (TypeError, ValueError):
                    pass

            if floor > 0:
                if max_tokens is None or (
                    isinstance(max_tokens, int)
                    and not isinstance(max_tokens, bool)
                    and max_tokens < floor
                ):
                    log.info(
                        "GeminiThinkingBudgetCallback: raising max_tokens %s -> %d "
                        "to preserve thinking budget for %s",
                        max_tokens, floor, model,
                    )
                    data["max_tokens"] = floor
                    return data
            break  # first matching rule wins

        return None


class TaskToolEnumInjectionCallback(CustomLogger):
    """
    Callback that injects subagent_type enum list into task tool parameters
    to prevent model hallucination. Disabled by default.
    """
    def __init__(self, enabled: bool = False) -> None:
        super().__init__()
        self.enabled = enabled

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[dict]:
        if not self.enabled:
            return None

        tools = data.get("tools")
        if isinstance(tools, list) and tools:
            copied_tools = copy.deepcopy(tools)
            rewrote = False
            for t in copied_tools:
                if isinstance(t, dict) and _rewrite_task_tool(t):
                    rewrote = True
            if rewrote:
                data["tools"] = copied_tools
                return data
        return None


class FallbackChainCommaRewriterCallback(CustomLogger):
    """
    Hook to rewrite comma-separated fallback chain models to semicolon-separated
    internal aliases before LiteLLM's route_request splits them on commas.
    """
    def __init__(self) -> None:
        super().__init__()

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[dict]:
        model = data.get("model")
        if isinstance(model, str):
            s = model
            if s.startswith("open-llm-proxy/"):
                s = s[len("open-llm-proxy/"):]
                
            if s.startswith("[") and s.endswith("]"):
                s = s.replace(",", ";")
                
            if s != model:
                log.info("Rewrote model name: %s -> %s", model, s)
                data["model"] = s
                return data
        return None


def _is_ollama_response(response: Any) -> bool:
    """True when the winning deployment is a local ollama hop.

    Detected via the served model id (``ollama_chat/...`` / ``ollama/...``)
    on the response or its hidden params. Scoped narrowly so we never touch
    remote provider responses.
    """
    candidates: list[str] = []
    model = getattr(response, "model", None)
    if isinstance(model, str):
        candidates.append(model)
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        for k in ("model", "custom_llm_provider"):
            v = hidden.get(k)
            if isinstance(v, str):
                candidates.append(v)
    return any("ollama" in c.lower() for c in candidates)


class OllamaReasoningStripCallback(CustomLogger):
    """Strip ``reasoning_content`` from local ollama (qwen3-vl) responses.

    Reasoning-model ollama deployments emit a large ``reasoning_content``
    block followed by a short final ``content``. opencode's
    ``@ai-sdk/openai-compatible`` client drops the final content in that
    shape, rendering an empty answer. We discard the reasoning so the client
    only sees the plain answer. Scoped to the ollama hop only; remote
    providers are untouched.
    """

    @staticmethod
    def _strip_message(msg: Any) -> None:
        if msg is None:
            return
        # Pydantic message objects and plain dicts both appear here.
        if isinstance(msg, dict):
            for k in ("reasoning_content", "reasoning", "thinking"):
                if msg.get(k):
                    msg[k] = None
        else:
            for k in ("reasoning_content", "reasoning", "thinking"):
                if getattr(msg, k, None):
                    try:
                        setattr(msg, k, None)
                    except Exception:
                        pass

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        try:
            if not _is_ollama_response(response):
                return response
            choices = getattr(response, "choices", None)
            if isinstance(choices, list):
                for ch in choices:
                    self._strip_message(getattr(ch, "message", None))
        except Exception:
            log.debug("OllamaReasoningStripCallback: non-stream strip skipped", exc_info=True)
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict,
    ):
        async for chunk in response:
            try:
                if _is_ollama_response(chunk):
                    choices = getattr(chunk, "choices", None)
                    if isinstance(choices, list):
                        for ch in choices:
                            self._strip_message(getattr(ch, "delta", None))
            except Exception:
                log.debug("OllamaReasoningStripCallback: stream strip skipped", exc_info=True)
            yield chunk


class ServedByCallback(CustomLogger):
    """Expose the concrete winning deployment in response headers."""

    async def async_post_call_response_headers_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: Any,
        request_headers: Optional[dict[str, str]] = None,
        litellm_call_info: Optional[dict[str, Any]] = None,
    ) -> dict[str, str]:
        key = served_by_from_data(data)

        if not key and isinstance(litellm_call_info, dict):
            model_info = litellm_call_info.get("model_info")
            if isinstance(model_info, dict):
                key = model_info.get("rate_limit_key")
            if not key:
                model_id = litellm_call_info.get("model_id")
                if isinstance(model_id, str):
                    try:
                        from litellm.proxy.proxy_server import llm_router
                        deployment = llm_router.get_model_info(id=model_id)
                        if isinstance(deployment, dict):
                            model_info = deployment.get("model_info")
                            if isinstance(model_info, dict):
                                key = model_info.get("rate_limit_key")
                    except Exception:
                        pass

        if not key and response is not None:
            hidden_params = getattr(response, "_hidden_params", None) or {}
            model_id = hidden_params.get("model_id") if isinstance(hidden_params, dict) else None
            if isinstance(model_id, str):
                try:
                    from litellm.proxy.proxy_server import llm_router
                    deployment = llm_router.get_model_info(id=model_id)
                    if isinstance(deployment, dict):
                        model_info = deployment.get("model_info")
                        if isinstance(model_info, dict):
                            key = model_info.get("rate_limit_key")
                except Exception:
                    pass

        if isinstance(key, str) and key:
            try:
                attr_id = attribution_id_from_headers(request_headers)
                if attr_id:
                    # Header lookup reports latest winner but does not consume
                    # the inline change announcement.
                    global_attribution_store.set(attr_id, key)
            except Exception:
                pass
            return {"x-open-llm-proxy-served-by": key}
        return {}


def _deployment_rate_limit_key(deployment: dict) -> Optional[str]:
    if not isinstance(deployment, dict):
        return None
    model_info = deployment.get("model_info")
    if not isinstance(model_info, dict):
        return None
    key = model_info.get("rate_limit_key")
    return key if isinstance(key, str) and "/" in key else None


class StickyRoutingCallback(CustomLogger):
    """Session affinity: pin a session to its last-warm deployment.

    Runs AFTER PersistentRateLimitCallback.async_filter_deployments, so the
    incoming healthy_deployments already exclude any provider on cooldown. If
    this session's previously-served deployment is still present, return ONLY
    it, keeping the provider prompt-prefix cache warm across turns. Otherwise
    return the list unchanged so normal fallback ordering picks the next model.

     Toggle via OPEN_LLM_PROXY_STICKY_ROUTING (default: off).
    """

    def __init__(self) -> None:
        super().__init__()
        self._enabled = os.environ.get(
            "OPEN_LLM_PROXY_STICKY_ROUTING", "0"
        ).lower() not in ("0", "false", "no")

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: Any,
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Any = None,
    ) -> list:
        if not self._enabled or not healthy_deployments or len(healthy_deployments) == 1:
            return healthy_deployments
        attr_id = None
        if isinstance(request_kwargs, dict):
            attr_id = attribution_id_from_data(request_kwargs)
        if not attr_id:
            return healthy_deployments
        preferred = global_attribution_store.get(attr_id)
        if not preferred:
            return healthy_deployments
        for deployment in healthy_deployments:
            if _deployment_rate_limit_key(deployment) == preferred:
                log.info(
                    "StickyRoutingCallback: pinning session %s to warm deployment %s",
                    attr_id, preferred,
                )
                return [deployment]
        return healthy_deployments
