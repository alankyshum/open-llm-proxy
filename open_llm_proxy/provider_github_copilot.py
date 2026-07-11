from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, AsyncIterator, Iterator, Optional

import httpx
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.utils import GenericStreamingChunk, ModelResponse

from open_llm_proxy import copilot_creds
from open_llm_proxy.errors import custom_rate_limit_error

log = logging.getLogger("open_llm_proxy.provider_github_copilot")


def _initiator_for(body: dict[str, Any]) -> str:
    msgs = body.get("messages") or []
    if not msgs:
        return "user"
    last = msgs[-1] if isinstance(msgs[-1], dict) else {}
    role = last.get("role")
    if role in ("tool", "assistant"):
        return "agent"
    return "user"


def _has_image_part(body: dict[str, Any]) -> bool:
    try:
        for m in body.get("messages") or []:
            if not isinstance(m, dict):
                continue
            c = m.get("content")
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        for item in body.get("input") or []:
            if not isinstance(item, dict):
                continue
            c = item.get("content")
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") in ("input_image", "image_url"):
                        return True
    except Exception:
        pass
    return False


def copilot_chat_to_responses(body: dict[str, Any]) -> dict[str, Any]:
    res = {
        k: v
        for k, v in body.items()
        if k
        not in (
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "tools",
            "tool_choice",
        )
    }

    if "model" in body:
        res["model"] = body["model"]

    if "tools" in body:
        res_tools = []
        for t in body["tools"]:
            if isinstance(t, dict):
                if t.get("type") == "function" and "function" in t:
                    f = t["function"]
                    res_tools.append({
                        "type": "function",
                        "name": f.get("name"),
                        "description": f.get("description"),
                        "parameters": f.get("parameters"),
                    })
                else:
                    res_tools.append(t)
            else:
                res_tools.append(t)
        res["tools"] = res_tools

    if "tool_choice" in body:
        tc = body["tool_choice"]
        if isinstance(tc, str):
            res["tool_choice"] = tc
        elif isinstance(tc, dict):
            if tc.get("type") == "function" and "function" in tc:
                f = tc["function"]
                res["tool_choice"] = {
                    "type": "function",
                    "name": f.get("name"),
                }
            else:
                res["tool_choice"] = tc

    if "messages" in body:
        input_items = []
        for m in body["messages"]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": m.get("tool_call_id"),
                    "output": m.get("content"),
                })
            elif role == "assistant" and m.get("tool_calls"):
                if m.get("content"):
                    input_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": m.get("content")}],
                    })
                for tc in m["tool_calls"]:
                    f = tc.get("function") or {}
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.get("id"),
                        "name": f.get("name"),
                        "arguments": f.get("arguments"),
                        "status": tc.get("status") or "completed",
                    })
            else:
                content = m.get("content")
                text_part_type = "output_text" if role == "assistant" else "input_text"
                if isinstance(content, str):
                    mapped_content = [{"type": text_part_type, "text": content}]
                elif isinstance(content, list):
                    mapped_content = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                mapped_content.append({
                                    "type": text_part_type,
                                    "text": part.get("text", ""),
                                })
                            else:
                                mapped_content.append(part)
                        else:
                            mapped_content.append(part)
                else:
                    mapped_content = content

                item = {
                    "role": role,
                    "content": mapped_content,
                }
                if role == "assistant":
                    item["type"] = "message"
                input_items.append(item)
        res["input"] = input_items

    if "max_completion_tokens" in body:
        res["max_output_tokens"] = body["max_completion_tokens"]
    elif "max_tokens" in body:
        res["max_output_tokens"] = body["max_tokens"]

    return res


def copilot_responses_to_chat(
    resp: dict[str, Any],
    *,
    completion_id: str,
    model: str,
    stream_requested: bool = False,
) -> dict[str, Any]:
    text_parts = []
    tool_calls = []
    is_truncated = False

    try:
        output = resp.get("output") or []
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                itype = item.get("type")
                if itype == "message":
                    content = item.get("content") or []
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "output_text":
                                t = part.get("text", "")
                                if t:
                                    text_parts.append(t)
                    if item.get("status") == "truncated":
                        is_truncated = True
                elif itype == "function_call":
                    cid = item.get("call_id")
                    name = item.get("name")
                    args = item.get("arguments")
                    tool_calls.append({
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args,
                        },
                    })
                    if item.get("status") == "truncated":
                        is_truncated = True
    except Exception:
        pass

    text = "".join(text_parts) if text_parts else ("" if not tool_calls else None)

    if tool_calls:
        finish_reason = "tool_calls"
    elif is_truncated or resp.get("incomplete_details") is not None:
        finish_reason = "length"
    else:
        finish_reason = "stop"

    message: dict[str, Any] = {
        "role": "assistant",
        "content": text,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    copilot_usage = resp.get("copilot_usage") or {}
    usage = {
        "prompt_tokens": copilot_usage.get("prompt_tokens", 0),
        "completion_tokens": copilot_usage.get("completion_tokens", 0),
        "total_tokens": copilot_usage.get("total_tokens", 0),
    }
    for k, v in copilot_usage.items():
        if k not in usage:
            usage[k] = v

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


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


class GithubCopilotLLM(CustomLLM):
    def __init__(self) -> None:
        self._clients_by_loop = {}
        self._client_lock = asyncio.Lock()
        self._endpoint_cache: Optional[tuple[float, dict[str, str]]] = None
        self._endpoint_ttl = 300.0

    async def _get_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if loop not in self._clients_by_loop:
            limits = httpx.Limits(max_keepalive_connections=32, max_connections=64)
            timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)
            self._clients_by_loop[loop] = httpx.AsyncClient(limits=limits, timeout=timeout, http2=False)
        return self._clients_by_loop[loop]

    @staticmethod
    def _strip_model_prefix(model: str) -> str:
        """Strip github-copilot/ and gh- prefix from model name."""
        s = model
        if s.startswith("github-copilot/"):
            s = s[len("github-copilot/"):]
        if s.startswith("gh-"):
            s = s[len("gh-"):]
        return s

    async def get_endpoint_for_model(self, model: str) -> str:
        model_str = self._strip_model_prefix(model)

        now = time.monotonic()
        if self._endpoint_cache and (now - self._endpoint_cache[0]) < self._endpoint_ttl:
            if model_str in self._endpoint_cache[1]:
                return self._endpoint_cache[1][model_str]

        try:
            token, base_url = await copilot_creds.get_copilot_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": copilot_creds.USER_AGENT,
                "Editor-Version": copilot_creds.EDITOR_VERSION,
                "Editor-Plugin-Version": copilot_creds.EDITOR_PLUGIN_VERSION,
                "Copilot-Integration-Id": copilot_creds.COPILOT_INTEGRATION_ID,
                "X-GitHub-Api-Version": copilot_creds.X_GITHUB_API_VERSION,
            }
            client = await self._get_client()
            r = await client.get(f"{base_url}/models", headers=headers)
            r.raise_for_status()
            data = r.json().get("data", [])
            model_map = {}
            for m in data:
                mid = m.get("id")
                if not mid:
                    continue
                endpoints = m.get("supported_endpoints")
                if isinstance(endpoints, list):
                    if "/chat/completions" in endpoints:
                        endpoint = "/chat/completions"
                    elif "/responses" in endpoints:
                        endpoint = "/responses"
                    else:
                        endpoint = self._heuristic_fallback(mid)
                else:
                    endpoint = self._heuristic_fallback(mid)
                model_map[mid] = endpoint

            self._endpoint_cache = (now, model_map)
            if model_str in model_map:
                return model_map[model_str]
        except Exception as e:
            log.debug("Failed to list models, fallback to heuristic: %s", e)

        return self._heuristic_fallback(model_str)

    def _heuristic_fallback(self, model: str) -> str:
        match = re.match(r"^gpt-(\d+)", model)
        if match:
            n = int(match.group(1))
            if n >= 5 and not model.startswith("gpt-5-mini"):
                return "/responses"
        return "/chat/completions"

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

        model_str = self._strip_model_prefix(model)

        try:
            token, base_url = await copilot_creds.get_copilot_token()
        except copilot_creds.CopilotAuthError:
            raise  # No status_code → litellm treats as retriable → triggers fallback
        except Exception as e:
            raise CustomLLMError(status_code=500, message=f"Copilot auth error: {e}")

        endpoint = await self.get_endpoint_for_model(model_str)

        body = {
            "model": model_str,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools
        if max_tokens:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": copilot_creds.USER_AGENT,
            "Editor-Version": copilot_creds.EDITOR_VERSION,
            "Editor-Plugin-Version": copilot_creds.EDITOR_PLUGIN_VERSION,
            "Copilot-Integration-Id": copilot_creds.COPILOT_INTEGRATION_ID,
            "Openai-Intent": "conversation-panel",
            "X-GitHub-Api-Version": copilot_creds.X_GITHUB_API_VERSION,
            "X-Request-Id": str(uuid.uuid4()),
        }
        headers["X-Initiator"] = _initiator_for(body)
        if _has_image_part(body):
            headers["Copilot-Vision-Request"] = "true"

        client = await self._get_client()

        async def send_with_retry(req_func):
            try:
                resp = await req_func()
                if resp.status_code == 401:
                    await resp.aclose() if hasattr(resp, "aclose") else None
                    copilot_creds.invalidate_short_lived()
                    new_token, _ = await copilot_creds.get_copilot_token()
                    headers["Authorization"] = f"Bearer {new_token}"
                    resp = await req_func()
                if resp.status_code == 429:
                    raise custom_rate_limit_error(
                        "Rate limited",
                        headers=dict(resp.headers),
                        rate_limit_origin_key=f"github-copilot/{model_str}",
                    )
                return resp
            except copilot_creds.CopilotAuthError:
                raise  # Let auth errors bubble up without 4xx status
            except CustomLLMError:
                raise
            except Exception as exc:
                raise CustomLLMError(status_code=500, message=str(exc))

        if endpoint == "/responses":
            # Translate request and treat as non-streaming JSON internally
            translated_body = copilot_chat_to_responses(body)
            headers["Accept"] = "application/json"

            async def _send_responses():
                req = client.build_request("POST", f"{base_url}/responses", json=translated_body, headers=headers)
                return await client.send(req)

            resp = await send_with_retry(_send_responses)
            if resp.status_code != 200:
                raise CustomLLMError(status_code=resp.status_code, message=resp.text)

            resp_data = resp.json()
            chat_dict = copilot_responses_to_chat(
                resp_data,
                completion_id=f"chatcmpl-{uuid.uuid4().hex}",
                model=model,
                stream_requested=True,
            )

            choices = chat_dict.get("choices", [])
            usage = chat_dict.get("usage", {})
            if choices:
                choice = choices[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])
                finish_reason = choice.get("finish_reason", "stop")

                if content:
                    yield chunk_to_generic({
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content}
                        }]
                    })
                if tool_calls:
                    yield chunk_to_generic({
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": tool_calls}
                        }]
                    })

                yield chunk_to_generic({
                    "choices": [{
                        "index": 0,
                        "delta": {}
                    }]
                }, is_finished=True, finish_reason=finish_reason)

        else:
            # /chat/completions true streaming
            body["stream"] = True
            headers["Accept"] = "text/event-stream"

            async def _send_chat():
                req = client.build_request("POST", f"{base_url}/chat/completions", json=body, headers=headers)
                return await client.send(req, stream=True)

            resp = await send_with_retry(_send_chat)
            if resp.status_code != 200:
                await resp.aclose()
                raise CustomLLMError(status_code=resp.status_code, message=f"HTTP {resp.status_code}")

            stream_finished = False
            saw_tool_use = False
            try:
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            ch = json.loads(data_str)
                            choices = ch.get("choices") or []
                            choice = choices[0] if choices else {}
                            finish_reason = choice.get("finish_reason")
                            delta = choice.get("delta") or {}
                            saw_tool_use = saw_tool_use or bool(delta.get("tool_calls"))
                            if finish_reason is not None:
                                stream_finished = True
                            yield chunk_to_generic(
                                ch,
                                is_finished=finish_reason is not None,
                                finish_reason=finish_reason or "",
                            )
                        except Exception:
                            pass
            finally:
                await resp.aclose()

            if not stream_finished:
                yield {
                    "text": "",
                    "tool_use": None,
                    "is_finished": True,
                    "finish_reason": "tool_calls" if saw_tool_use else "stop",
                    "usage": None,
                    "index": 0,
                }

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

        model_str = self._strip_model_prefix(model)

        try:
            token, base_url = await copilot_creds.get_copilot_token()
        except copilot_creds.CopilotAuthError:
            raise  # No status_code → litellm treats as retriable → triggers fallback
        except Exception as e:
            raise CustomLLMError(status_code=500, message=f"Copilot auth error: {e}")

        endpoint = await self.get_endpoint_for_model(model_str)

        body = {
            "model": model_str,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools
        if max_tokens:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": copilot_creds.USER_AGENT,
            "Editor-Version": copilot_creds.EDITOR_VERSION,
            "Editor-Plugin-Version": copilot_creds.EDITOR_PLUGIN_VERSION,
            "Copilot-Integration-Id": copilot_creds.COPILOT_INTEGRATION_ID,
            "Openai-Intent": "conversation-panel",
            "X-GitHub-Api-Version": copilot_creds.X_GITHUB_API_VERSION,
            "X-Request-Id": str(uuid.uuid4()),
        }
        headers["X-Initiator"] = _initiator_for(body)
        if _has_image_part(body):
            headers["Copilot-Vision-Request"] = "true"

        client = await self._get_client()

        async def send_with_retry(req_func):
            try:
                resp = await req_func()
                if resp.status_code == 401:
                    await resp.aclose() if hasattr(resp, "aclose") else None
                    copilot_creds.invalidate_short_lived()
                    new_token, _ = await copilot_creds.get_copilot_token()
                    headers["Authorization"] = f"Bearer {new_token}"
                    resp = await req_func()
                if resp.status_code == 429:
                    raise custom_rate_limit_error(
                        "Rate limited",
                        headers=dict(resp.headers),
                        rate_limit_origin_key=f"github-copilot/{model_str}",
                    )
                return resp
            except copilot_creds.CopilotAuthError:
                raise  # Let auth errors bubble up without 4xx status
            except CustomLLMError:
                raise
            except Exception as exc:
                raise CustomLLMError(status_code=500, message=str(exc))

        if endpoint == "/responses":
            translated_body = copilot_chat_to_responses(body)
            headers["Accept"] = "application/json"

            async def _send_responses():
                req = client.build_request("POST", f"{base_url}/responses", json=translated_body, headers=headers)
                return await client.send(req)

            resp = await send_with_retry(_send_responses)
            if resp.status_code != 200:
                raise CustomLLMError(status_code=resp.status_code, message=resp.text)

            resp_data = resp.json()
            chat_dict = copilot_responses_to_chat(
                resp_data,
                completion_id=f"chatcmpl-{uuid.uuid4().hex}",
                model=model,
                stream_requested=False,
            )

            choices = chat_dict.get("choices", [])
            usage = chat_dict.get("usage", {})
            return ModelResponse(
                id=chat_dict.get("id"),
                choices=choices,
                model=chat_dict.get("model"),
                usage=usage,
            )

        else:
            headers["Accept"] = "application/json"

            async def _send_chat():
                req = client.build_request("POST", f"{base_url}/chat/completions", json=body, headers=headers)
                return await client.send(req)

            resp = await send_with_retry(_send_chat)
            if resp.status_code != 200:
                raise CustomLLMError(status_code=resp.status_code, message=resp.text)

            resp_json = resp.json()
            choices = resp_json.get("choices", [])
            usage = resp_json.get("usage", {})
            return ModelResponse(
                id=resp_json.get("id"),
                choices=choices,
                model=model,
                usage=usage,
            )


copilot_handler = GithubCopilotLLM()
