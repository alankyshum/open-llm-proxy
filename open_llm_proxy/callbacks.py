from typing import Any, Dict, List, Optional
import os
import re
import logging
import copy
from litellm.integrations.custom_logger import CustomLogger

log = logging.getLogger("open_llm_proxy.callbacks")

_SERVED_BY_PREFIX = "[served-by: "


def is_first_assistant_turn(data: dict[str, Any]) -> bool:
    """Return whether request history has no completed assistant turn yet."""
    messages = data.get("messages") or []
    return not any(
        (message.get("role") if isinstance(message, dict) else getattr(message, "role", None))
        == "assistant"
        for message in messages
    )


def served_by_banner(data: dict[str, Any]) -> str | None:
    """Return attribution for the concrete deployment selected by LiteLLM."""
    model_info = data.get("model_info") or {}
    if not model_info:
        litellm_params = data.get("litellm_params") or {}
        model_info = litellm_params.get("model_info") or {}
    key = model_info.get("rate_limit_key") if isinstance(model_info, dict) else None
    if not isinstance(key, str) or "/" not in key:
        return None
    return f"{_SERVED_BY_PREFIX}{key}]\n\n"


def served_by_response_banner(response: Any, router: Any) -> str | None:
    """Resolve the selected deployment from LiteLLM's response metadata."""
    hidden_params = getattr(response, "_hidden_params", None) or {}
    model_id = hidden_params.get("model_id") if isinstance(hidden_params, dict) else None
    if not isinstance(model_id, str):
        return None
    deployment = router.get_model_info(id=model_id)
    return served_by_banner(deployment) if isinstance(deployment, dict) else None


def prefix_response_content(response: Any, banner: str) -> Any:
    """Prefix OpenAI-compatible assistant content without disturbing tool calls."""
    choices = getattr(response, "choices", None)
    if not choices:
        return response
    message = getattr(choices[0], "message", None)
    if message is None:
        return response
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.startswith(_SERVED_BY_PREFIX):
        return response
    message.content = banner + (content or "")
    return response

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
            ("gpt-5.5",                 4096, None),
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


class ServedByCallback(CustomLogger):
    """Expose the concrete winning deployment in non-streaming responses."""

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        if not is_first_assistant_turn(data):
            return response
        banner = served_by_banner(data)
        if banner is None:
            return response
        return prefix_response_content(response, banner)
