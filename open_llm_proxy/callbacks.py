from typing import Any, Dict, List, Optional
import os
import re
import logging
import copy
from litellm.integrations.custom_logger import CustomLogger

log = logging.getLogger("open_llm_proxy.callbacks")

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
            ("gemini",              1024, "KILO_PROXY_GOOGLE_MIN_MAX_TOKENS"),
            ("google/",             1024, "KILO_PROXY_GOOGLE_MIN_MAX_TOKENS"),
            ("deepseek-v4-flash-free",  4096, None),
            ("nemotron-3-ultra-free",   4096, None),
        ]

    async def async_pre_request_hook(
        self, model: str, messages: List, kwargs: Dict
    ) -> Optional[Dict]:
        model_lower = model.lower()
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
                mt = kwargs.get("max_tokens")
                if isinstance(mt, int) and not isinstance(mt, bool) and mt < floor:
                    log.info(
                        "GeminiThinkingBudgetCallback: raising max_tokens %d -> %d "
                        "to preserve thinking budget for %s",
                        mt, floor, model,
                    )
                    kwargs["max_tokens"] = floor
                    return kwargs
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

    async def async_pre_request_hook(
        self, model: str, messages: List, kwargs: Dict
    ) -> Optional[Dict]:
        if not self.enabled:
            return None
            
        tools = kwargs.get("tools")
        if isinstance(tools, list) and tools:
            copied_tools = copy.deepcopy(tools)
            rewrote = False
            for t in copied_tools:
                if isinstance(t, dict) and _rewrite_task_tool(t):
                    rewrote = True
            if rewrote:
                kwargs["tools"] = copied_tools
                return kwargs
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
            elif s.startswith("kilo-proxy/"):
                s = s[len("kilo-proxy/"):]
                
            if s.startswith("[") and s.endswith("]"):
                s = s.replace(",", ";")
                
            if s != model:
                log.info("Rewrote model name: %s -> %s", model, s)
                data["model"] = s
                return data
        return None
