from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Iterator

from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.utils import GenericStreamingChunk, ModelResponse

from open_llm_proxy import anthropic_client, translator


def chunk_to_generic(ch: dict[str, Any], is_finished: bool = False, finish_reason: str = "") -> GenericStreamingChunk:
    choices = ch.get("choices", [])
    text = ""
    tool_use = None
    idx = 0
    if choices:
        choice = choices[0]
        idx = choice.get("index", 0)
        delta = choice.get("delta", {})
        text = delta.get("content", "") or ""
        tcalls = delta.get("tool_calls", None)
        if tcalls and len(tcalls) > 0:
            tool_use = tcalls[0]
    
    chunk: GenericStreamingChunk = {
        "text": text,
        "is_finished": is_finished,
        "finish_reason": finish_reason,
        "usage": None,
        "index": idx,
    }
    if tool_use is not None:
        chunk["tool_use"] = tool_use
    return chunk


def anthropic_response_to_model_response(
    raw: dict[str, Any],
    model: str,
) -> ModelResponse:
    content_parts = []
    tool_calls = []
    for block in raw.get("content", []):
        if block.get("type") == "text":
            content_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            args_in = block.get("input", {})
            transformed = translator.apply_rtk_to_args(args_in) if isinstance(args_in, dict) else args_in
            args_str = json.dumps(transformed, separators=(",", ":"))
            tool_calls.append({
                "id": block.get("id"),
                "type": "function",
                "function": {
                    "name": block.get("name"),
                    "arguments": args_str,
                }
            })
    
    content_str = "".join(content_parts)
    stop_reason = raw.get("stop_reason") or "end_turn"
    finish_reason = translator._ANTHROPIC_FINISH_MAP.get(stop_reason, "stop")
    
    usage = raw.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens", 0))
    completion_tokens = int(usage.get("output_tokens", 0))
    total_tokens = prompt_tokens + completion_tokens
    
    return ModelResponse(
        id=f"chatcmpl-{raw.get('id', uuid.uuid4().hex)}",
        choices=[
            {
                "finish_reason": finish_reason,
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content_str if content_str else None,
                    "tool_calls": tool_calls if tool_calls else None,
                }
            }
        ],
        model=model,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    )


class ClaudeCliLLM(CustomLLM):
    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        async_gen = self.astreaming(*args, **kwargs)
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        queue = asyncio.Queue()
        
        async def producer():
            try:
                async for item in async_gen:
                    await queue.put((False, item))
            except Exception as e:
                await queue.put((True, e))
            finally:
                await queue.put((True, None))
                
        task = loop.create_task(producer())
        
        while True:
            try:
                is_done, val = loop.run_until_complete(queue.get())
            except Exception as e:
                raise e
            if is_done:
                if isinstance(val, Exception):
                    raise val
                break
            yield val

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        model = kwargs.get("model") or (args[0] if len(args) > 0 else "")
        messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
        optional_params = kwargs.get("optional_params") or {}
        
        tools = kwargs.get("tools") or optional_params.get("tools")
        max_tokens = kwargs.get("max_tokens") or optional_params.get("max_tokens")
        temperature = kwargs.get("temperature") or optional_params.get("temperature")
        
        model_str = model
        if model_str.startswith("claude-cli/"):
            model_str = model_str[len("claude-cli/"):]
            
        base_model, thinking_level = translator.parse_model(model_str)
        
        payload = translator.build_anthropic_payload(
            model=base_model,
            openai_messages=messages,
            openai_tools=tools,
            thinking_level=thinking_level,
            max_tokens=max_tokens or translator.default_max_tokens_for_model(base_model),
            temperature=temperature,
            stream=True,
        )
        
        state: dict[str, Any] = {
            "role_emitted": False,
            "next_tool_idx": 0,
            "block_to_tool_idx": {},
            "block_arg_buf": {},
            "block_to_tool_use_id": {},
            "block_to_tool_name": {},
            "finish_reason": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        
        try:
            async for event_name, event_data in anthropic_client.stream_messages(payload):
                chunks = translator.anthropic_event_to_openai_chunks(
                    event_name, event_data,
                    completion_id=completion_id, model=base_model, state=state,
                )
                for ch in chunks:
                    yield chunk_to_generic(ch, is_finished=False, finish_reason="")
                    
            finish = state.get("finish_reason") or "stop"
            usage_block = {
                "prompt_tokens": state["usage"].get("prompt_tokens", 0),
                "completion_tokens": state["usage"].get("completion_tokens", 0),
                "total_tokens": state["usage"].get("prompt_tokens", 0) + state["usage"].get("completion_tokens", 0)
            }
            yield {
                "text": "",
                "tool_use": None,
                "is_finished": True,
                "finish_reason": finish,
                "usage": usage_block,
                "index": 0,
            }
        except Exception as e:
            from open_llm_proxy.errors import map_rate_limit_error
            raise map_rate_limit_error(e)

    def completion(self, *args, **kwargs) -> ModelResponse:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.acompletion(*args, **kwargs))

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        model = kwargs.get("model") or (args[0] if len(args) > 0 else "")
        messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
        optional_params = kwargs.get("optional_params") or {}
        
        tools = kwargs.get("tools") or optional_params.get("tools")
        max_tokens = kwargs.get("max_tokens") or optional_params.get("max_tokens")
        temperature = kwargs.get("temperature") or optional_params.get("temperature")
        
        model_str = model
        if model_str.startswith("claude-cli/"):
            model_str = model_str[len("claude-cli/"):]
            
        base_model, thinking_level = translator.parse_model(model_str)
        
        payload = translator.build_anthropic_payload(
            model=base_model,
            openai_messages=messages,
            openai_tools=tools,
            thinking_level=thinking_level,
            max_tokens=max_tokens or translator.default_max_tokens_for_model(base_model),
            temperature=temperature,
            stream=False,
        )
        
        try:
            raw_response = await anthropic_client.send_messages(payload)
            return anthropic_response_to_model_response(raw_response, model)
        except Exception as e:
            from open_llm_proxy.errors import map_rate_limit_error
            raise map_rate_limit_error(e)


claude_cli_handler = ClaudeCliLLM()
