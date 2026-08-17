from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any

from open_llm_proxy.errors import TranslationError

# ── Model catalogue ──────────────────────────────────────────────────────────

_MODEL_CATALOG: list[dict] = [
    {
        "id": "claude-opus-4-8",
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "description": "Most capable Opus 4.8",
        "default_thinking": "xhigh",
        "owned_by": "anthropic",
    },
    {
        "id": "claude-sonnet-5",
        "context_window": 1_000_000,
        "max_output_tokens": 64_000,
        "description": "Sonnet 5 — balanced (recommended)",
        "default_thinking": "high",
        "owned_by": "anthropic",
    },
    {
        "id": "claude-fable-5",
        "context_window": 200_000,
        "max_output_tokens": 64_000,
        "description": "Fable 5",
        "default_thinking": "none",
        "owned_by": "anthropic",
    },
    {
        "id": "claude-opus-4-7",
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "description": "Most capable — complex reasoning and agentic coding",
        "default_thinking": "xhigh",
        "owned_by": "anthropic",
    },
    {
        "id": "claude-sonnet-4-6",
        "context_window": 1_000_000,
        "max_output_tokens": 64_000,
        "description": "Balanced speed and intelligence (recommended)",
        "default_thinking": "high",
        "owned_by": "anthropic",
    },
    {
        "id": "claude-haiku-4-5",
        "context_window": 200_000,
        "max_output_tokens": 64_000,
        "description": "Fastest — low-latency tasks",
        "default_thinking": "none",
        "owned_by": "anthropic",
    },
]

_THINKING_VARIANTS: list[tuple[str, str]] = [
    ("none", "no thinking"),
    ("low", "light thinking"),
    ("medium", "moderate thinking"),
    ("high", "deep thinking"),
    ("xhigh", "max thinking"),
]

_MODEL_NORMALIZE: list[tuple[str, str]] = [
    (r"^claude-haiku-4-5-\d+$", "claude-haiku-4-5"),
    (r"^claude-opus-4-[56](?:\[1m\])?$", "claude-opus-4-7"),
]


def _normalize_model_id(raw: str) -> str | None:
    raw_norm = re.sub(r"^(claude-[a-z0-9-]+-\d+)\.(\d+)$", r"\1-\2", raw)
    for pattern, canonical in _MODEL_NORMALIZE:
        if re.match(pattern, raw_norm):
            return canonical
    catalog_ids = {m["id"] for m in _MODEL_CATALOG}
    if raw_norm in catalog_ids:
        return raw_norm
    return None


def _discover_models_from_claude_json() -> list[dict]:
    from pathlib import Path

    path = Path.home() / ".claude.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    seen: set[str] = set()
    for proj in (data.get("projects") or {}).values():
        for model_id in proj.get("lastModelUsage") or {}:
            canonical = _normalize_model_id(model_id)
            if canonical:
                seen.add(canonical)

    for entry in data.get("additionalModelOptionsCache") or []:
        if isinstance(entry, dict) and entry.get("id"):
            canonical = _normalize_model_id(entry["id"])
            if canonical:
                seen.add(canonical)

    known_ids = {m["id"] for m in _MODEL_CATALOG}
    extras = []
    for model_id in sorted(seen - known_ids):
        extras.append(
            {
                "id": model_id,
                "context_window": 200_000,
                "max_output_tokens": 32_000,
                "description": "Discovered from claude CLI usage history",
                "default_thinking": "none",
                "owned_by": "anthropic",
            }
        )
    return extras


def build_models_list() -> list[dict]:
    catalog = _MODEL_CATALOG + _discover_models_from_claude_json()
    entries: list[dict] = []

    for spec in catalog:
        mid = spec["id"]
        ctx = spec["context_window"]
        max_out = spec["max_output_tokens"]
        base_desc = spec["description"]
        default_thinking = spec["default_thinking"]
        owned_by = spec.get("owned_by", "anthropic")

        default_label = next(
            (label for name, label in _THINKING_VARIANTS if name == default_thinking),
            default_thinking,
        )
        entries.append(
            {
                "id": mid,
                "object": "model",
                "owned_by": owned_by,
                "context_window": ctx,
                "max_output_tokens": max_out,
                "description": f"{base_desc} [default: {default_label}]",
            }
        )

        for variant_name, variant_label in _THINKING_VARIANTS:
            entries.append(
                {
                    "id": f"{mid}:{variant_name}",
                    "object": "model",
                    "owned_by": owned_by,
                    "context_window": ctx,
                    "max_output_tokens": max_out,
                    "description": f"{base_desc} [{variant_label}]",
                }
            )

    return entries


def translate_anthropic_models(anthropic_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for am in anthropic_models:
        mid = am.get("id")
        if not mid:
            continue
        display_name = am.get("display_name") or mid

        ctx = _MODEL_CONTEXT_WINDOWS.get(mid)
        if ctx is None:
            if "opus" in mid.lower():
                ctx = 1_000_000
            elif "sonnet" in mid.lower():
                ctx = 1_000_000
            elif "haiku" in mid.lower():
                ctx = 200_000
            else:
                ctx = 200_000

        max_out = default_max_tokens_for_model(mid)
        if max_out == 8192:
            if "opus" in mid.lower():
                max_out = 128_000
            elif "sonnet" in mid.lower():
                max_out = 64_000
            elif "haiku" in mid.lower():
                max_out = 64_000

        default_thinking = _DEFAULT_THINKING_LEVEL.get(mid)
        if default_thinking is None:
            if "opus" in mid.lower():
                default_thinking = "xhigh"
            elif "sonnet" in mid.lower():
                default_thinking = "high"
            else:
                default_thinking = "none"

        default_label = next(
            (label for name, label in _THINKING_VARIANTS if name == default_thinking),
            default_thinking,
        )

        entries.append(
            {
                "id": mid,
                "object": "model",
                "owned_by": "anthropic",
                "context_window": ctx,
                "max_output_tokens": max_out,
                "description": f"{display_name} [default: {default_label}]",
            }
        )

        for variant_name, variant_label in _THINKING_VARIANTS:
            entries.append(
                {
                    "id": f"{mid}:{variant_name}",
                    "object": "model",
                    "owned_by": "anthropic",
                    "context_window": ctx,
                    "max_output_tokens": max_out,
                    "description": f"{display_name} [{variant_label}]",
                }
            )

    return entries


_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-8": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-fable-5": 200_000,
    "claude-opus-4-7": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-haiku-4-5": 200_000,
}

_DEFAULT_THINKING_LEVEL: dict[str, str] = {
    "claude-opus-4-8": "xhigh",
    "claude-sonnet-5": "high",
    "claude-fable-5": "none",
    "claude-opus-4-7": "xhigh",
    "claude-sonnet-4-6": "high",
    "claude-haiku-4-5": "none",
}

_THINKING_BUDGETS: dict[str, int] = {
    "low": 1024,
    "medium": 5000,
    "high": 16000,
    "xhigh": 32000,
}


def context_window_for_model(model: str) -> int:
    base = model.split(":")[0] if ":" in model else model
    val = _MODEL_CONTEXT_WINDOWS.get(base)
    if val is not None:
        return val
    if "opus" in base.lower():
        return 1_000_000
    if "sonnet" in base.lower():
        return 1_000_000
    if "haiku" in base.lower():
        return 200_000
    return 200_000


def parse_model(model_str: str) -> tuple[str, str]:
    if ":" in model_str:
        base, variant = model_str.split(":", 1)
        base = base.strip()
        variant = variant.strip().lower()
    else:
        base = model_str.strip()
        variant = None

    base = re.sub(r"^(claude-[a-z0-9-]+-\d+)\.(\d+)$", r"\1-\2", base)

    if variant is None:
        level = _DEFAULT_THINKING_LEVEL.get(base, "none")
    else:
        level = variant
    return base, level


def thinking_budget(level: str) -> int | None:
    return _THINKING_BUDGETS.get(level.lower())


def default_max_tokens_for_model(model: str) -> int:
    base = model.split(":")[0] if ":" in model else model
    for spec in _MODEL_CATALOG:
        if spec["id"] == base:
            return spec["max_output_tokens"]
    if "opus" in base.lower():
        return 128_000
    if "sonnet" in base.lower():
        return 64_000
    if "haiku" in base.lower():
        return 64_000
    return 64_000


# ---------- inbound: OpenAI -> Anthropic Messages ----------


def extract_system_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        if m.get("role") != "system":
            continue
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(blk.get("text", ""))
    return "\n\n".join(p for p in parts if p)


def _content_to_text(content: Any, *, _depth: int = 0) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if _depth >= 16:
        return content if isinstance(content, str) else json.dumps(content, default=str)
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            t = part.get("type")
            if t == "text":
                out.append(part.get("text", ""))
            elif t in ("output_text", "input_text"):
                out.append(part.get("text", ""))
            elif t in ("tool-result", "tool_result"):
                output = part.get("output")
                if isinstance(output, dict):
                    val = output.get("value")
                    if isinstance(val, str):
                        out.append(val)
                    elif val is not None:
                        out.append(json.dumps(val))
                    else:
                        out.append("")
                elif "result" in part:
                    res = part.get("result")
                    if isinstance(res, str):
                        out.append(res)
                    else:
                        out.append(json.dumps(res))
                elif "content" in part:
                    inner = part.get("content")
                    if isinstance(inner, list):
                        out.append(_content_to_text(inner, _depth=_depth + 1))
                    elif isinstance(inner, str):
                        out.append(inner)
                    else:
                        out.append(str(inner) if inner is not None else "")
                else:
                    out.append("")
            elif t == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                out.append(f"[image: {url[:60]}...]")
            else:
                out.append(json.dumps(part))
        return "\n".join(out)
    return str(content)


def openai_tools_to_anthropic(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tools or []:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        entry: dict[str, Any] = {"name": name, "input_schema": params}
        desc = fn.get("description")
        if desc:
            entry["description"] = desc
        out.append(entry)
    return out


def _parse_data_uri(url: str) -> tuple[str, str] | None:
    if not url.startswith("data:"):
        return None
    try:
        header, b64 = url.split(",", 1)
    except ValueError:
        return None
    meta = header[len("data:") :]
    if ";base64" not in meta:
        return None
    media_type = meta.split(";")[0] or "image/png"
    return media_type, b64


def _user_content_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                blocks.append({"type": "text", "text": str(part)})
                continue
            t = part.get("type")
            if t == "text":
                txt = part.get("text", "")
                if txt:
                    blocks.append({"type": "text", "text": txt})
            elif t == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                parsed = _parse_data_uri(url)
                if parsed:
                    media_type, b64 = parsed
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64},
                        }
                    )
                else:
                    blocks.append({"type": "text", "text": f"[image: {url[:60]}...]"})
            elif t in ("tool_result", "tool-result"):
                tu_id = (
                    part.get("tool_use_id")
                    or part.get("toolCallId")
                    or part.get("tool_call_id")
                    or ""
                )
                if not tu_id:
                    raise TranslationError(
                        "inline tool_result block missing tool_use_id/toolCallId"
                    )
                text = _content_to_text([part])
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu_id,
                        "content": text,
                    }
                )
            else:
                blocks.append({"type": "text", "text": _content_to_text([part])})
        return blocks
    return [{"type": "text", "text": str(content)}]


def openai_messages_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system_text = extract_system_prompt(messages)
    system_blocks: list[dict[str, Any]] = (
        [{"type": "text", "text": system_text}] if system_text else []
    )

    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            blocks = _user_content_blocks(m.get("content"))
            if not blocks:
                continue
            out.append({"role": "user", "content": blocks})
        elif role == "assistant":
            blocks = []
            text = _content_to_text(m.get("content"))
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args_raw = fn.get("arguments")
                if isinstance(args_raw, str):
                    try:
                        args_in = json.loads(args_raw) if args_raw else {}
                    except json.JSONDecodeError:
                        args_in = {}
                elif isinstance(args_raw, dict):
                    args_in = args_raw
                else:
                    args_in = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id") or "",
                        "name": fn.get("name") or "",
                        "input": args_in,
                    }
                )
            if not blocks:
                continue
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            tcid = m.get("tool_call_id") or ""
            content = m.get("content")
            if not tcid and isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    cand = part.get("toolCallId") or part.get("tool_call_id")
                    if cand:
                        tcid = cand
                        break
            if not tcid:
                raise TranslationError("tool message missing tool_call_id/toolCallId")
            text = _content_to_text(content)
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tcid,
                            "content": text,
                        }
                    ],
                }
            )

    merged: list[dict[str, Any]] = []
    for msg in out:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append({"role": msg["role"], "content": list(msg["content"])})

    return system_blocks, merged


_DEFAULT_BILLING_HEADER = (
    "x-anthropic-billing-header: cc_version=2.1.126.824; cc_entrypoint=sdk-cli; cch=6b7d9;"
)


def billing_header_text() -> str:
    return os.environ.get("OPEN_LLM_PROXY_BILLING_HEADER", _DEFAULT_BILLING_HEADER)


def _billing_header_block() -> dict[str, Any]:
    return {"type": "text", "text": billing_header_text()}


def build_anthropic_payload(
    *,
    model: str,
    openai_messages: list[dict[str, Any]],
    openai_tools: list[dict[str, Any]] | None,
    thinking_level: str,
    max_tokens: int,
    temperature: Any,
    stream: bool,
) -> dict[str, Any]:
    base_model = model.split(":")[0] if ":" in model else model
    system_blocks, anth_msgs = openai_messages_to_anthropic(openai_messages)
    tools = openai_tools_to_anthropic(openai_tools)
    budget = thinking_budget(thinking_level)

    payload: dict[str, Any] = {
        "model": base_model,
        "messages": anth_msgs,
        "max_tokens": max_tokens,
        "stream": stream,
        "system": [_billing_header_block(), *system_blocks],
    }
    if tools:
        payload["tools"] = tools
    if budget is not None:
        payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
    else:
        if isinstance(temperature, (int, float)):
            payload["temperature"] = float(temperature)
    return payload


# ---------- outbound: OpenAI SSE chunk builders ----------


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _base_chunk(completion_id: str, model: str) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
    }


def text_delta_chunk(
    completion_id: str, model: str, text: str, *, first: bool = False
) -> dict[str, Any]:  # intentional long protocol text or compatibility message
    delta: dict[str, Any] = {"content": text}
    if first:
        delta["role"] = "assistant"
    c = _base_chunk(completion_id, model)
    c["choices"] = [{"index": 0, "delta": delta, "finish_reason": None}]
    return c


def role_only_chunk(completion_id: str, model: str) -> dict[str, Any]:
    c = _base_chunk(completion_id, model)
    c["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    return c


def tool_call_start_chunk(
    completion_id: str,
    model: str,
    *,
    index: int,
    tool_call_id: str,
    name: str,
) -> dict[str, Any]:
    c = _base_chunk(completion_id, model)
    c["choices"] = [
        {
            "index": 0,
            "delta": {
                "tool_calls": [
                    {
                        "index": index,
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }
                ]
            },
            "finish_reason": None,
        }
    ]
    return c


def tool_call_args_delta_chunk(
    completion_id: str,
    model: str,
    *,
    index: int,
    partial_args: str,
) -> dict[str, Any]:
    c = _base_chunk(completion_id, model)
    c["choices"] = [
        {
            "index": 0,
            "delta": {
                "tool_calls": [
                    {
                        "index": index,
                        "function": {"arguments": partial_args},
                    }
                ]
            },
            "finish_reason": None,
        }
    ]
    return c


def finish_chunk(completion_id: str, model: str, reason: str) -> dict[str, Any]:
    c = _base_chunk(completion_id, model)
    c["choices"] = [{"index": 0, "delta": {}, "finish_reason": reason}]
    return c


def usage_chunk(
    completion_id: str,
    model: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    context_window: int | None = None,
) -> dict[str, Any]:
    c = _base_chunk(completion_id, model)
    c["choices"] = []
    c["usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if context_window is not None:
        c["usage"]["context_window"] = context_window
    return c


# ---------- RTK transform ----------

_RTK_GIT_PREFIX = "git "


def apply_rtk_to_args(args: dict) -> dict:
    if os.environ.get("OPEN_LLM_PROXY_RTK_GIT") != "1":
        return args
    cmd = args.get("command")
    if not isinstance(cmd, str):
        return args
    stripped = cmd.lstrip()
    if stripped.startswith(_RTK_GIT_PREFIX):
        return {**args, "command": f"rtk {stripped}"}
    return args


# ---------- Anthropic SSE event → OpenAI chunks ----------

_ANTHROPIC_FINISH_MAP = {
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "end_turn": "stop",
    "stop_sequence": "stop",
}


def anthropic_event_to_openai_chunks(
    event_name: str,
    event_data: dict[str, Any],
    *,
    completion_id: str,
    model: str,
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    if event_name == "message_start":
        msg = event_data.get("message") or {}
        usage = msg.get("usage") or {}
        if "input_tokens" in usage:
            state["usage"]["prompt_tokens"] = int(usage["input_tokens"])
        if "output_tokens" in usage:
            state["usage"]["completion_tokens"] = int(usage["output_tokens"])
        if not state.get("role_emitted"):
            out.append(role_only_chunk(completion_id, model))
            state["role_emitted"] = True
        return out

    if event_name == "content_block_start":
        block = event_data.get("content_block") or {}
        idx = event_data.get("index", 0)
        btype = block.get("type")
        if btype == "tool_use":
            tool_idx = state["next_tool_idx"]
            state["next_tool_idx"] = tool_idx + 1
            state["block_to_tool_idx"][idx] = tool_idx
            state["block_to_tool_use_id"][idx] = block.get("id", "")
            state["block_to_tool_name"][idx] = block.get("name", "")
            state["block_arg_buf"][idx] = ""
            out.append(
                tool_call_start_chunk(
                    completion_id,
                    model,
                    index=tool_idx,
                    tool_call_id=block.get("id", ""),
                    name=block.get("name", ""),
                )
            )
        return out

    if event_name == "content_block_delta":
        idx = event_data.get("index", 0)
        delta = event_data.get("delta") or {}
        dtype = delta.get("type")
        if dtype == "text_delta":
            text = delta.get("text", "")
            if text:
                out.append(text_delta_chunk(completion_id, model, text))
        elif dtype == "input_json_delta":
            partial = delta.get("partial_json", "")
            if idx in state["block_arg_buf"]:
                state["block_arg_buf"][idx] += partial
        return out

    if event_name == "content_block_stop":
        idx = event_data.get("index", 0)
        if idx in state["block_to_tool_idx"]:
            tool_idx = state["block_to_tool_idx"][idx]
            buf = state["block_arg_buf"].get(idx, "")
            try:
                args_in = json.loads(buf) if buf else {}
            except json.JSONDecodeError:
                args_in = {}
            transformed = apply_rtk_to_args(args_in) if isinstance(args_in, dict) else args_in
            args_str = json.dumps(transformed, separators=(",", ":"))
            out.append(
                tool_call_args_delta_chunk(
                    completion_id,
                    model,
                    index=tool_idx,
                    partial_args=args_str,
                )
            )
        return out

    if event_name == "message_delta":
        delta = event_data.get("delta") or {}
        usage = event_data.get("usage") or {}
        if "output_tokens" in usage:
            state["usage"]["completion_tokens"] = int(usage["output_tokens"])
        if "input_tokens" in usage:
            state["usage"]["prompt_tokens"] = int(usage["input_tokens"])
        stop = delta.get("stop_reason")
        if stop:
            state["finish_reason"] = _ANTHROPIC_FINISH_MAP.get(stop, "stop")
        return out

    return out
