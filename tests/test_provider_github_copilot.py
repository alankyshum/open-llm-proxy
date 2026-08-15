from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from litellm.llms.custom_llm import CustomLLMError
from litellm.utils import ModelResponse

from open_llm_proxy import copilot_creds
from open_llm_proxy.provider_github_copilot import (
    GithubCopilotLLM,
    copilot_handler,
    _initiator_for,
    _has_image_part,
    _normalize_tools_for_copilot,
    _ensure_object_schemas_have_properties,
    copilot_chat_to_responses,
    copilot_responses_to_chat,
    _send_with_429_retry,
    _log_429_diagnostic,
    _redact_429_diagnostic,
)


def resolve_copilot_creds() -> bool:
    try:
        # Avoid running full asyncio loop in pytest setup block if not needed
        # Just check if we can get the oauth token
        return bool(copilot_creds.get_oauth_token())
    except Exception:
        return False


has_copilot = resolve_copilot_creds()


# ── OFFLINE UNIT TESTS ────────────────────────────────────────────────────────


def test_endpoint_heuristic_routing():
    handler = GithubCopilotLLM()
    assert handler._heuristic_fallback("gpt-5.5") == "/responses"
    assert handler._heuristic_fallback("gpt-5-mini") == "/chat/completions"
    assert handler._heuristic_fallback("claude-sonnet-5") == "/chat/completions"
    assert handler._heuristic_fallback("gemini-2.5-pro") == "/chat/completions"


@pytest.mark.anyio
async def test_429_retries_once_then_returns_second_429(monkeypatch):
    first = httpx.Response(
        429,
        headers={"retry-after": "0.01", "x-request-id": "req-1"},
        stream=httpx.ByteStream(b"temporary"),
        request=httpx.Request("POST", "https://api.example.test"),
    )
    second = httpx.Response(
        429,
        headers={"retry-after": "0.01"},
        stream=httpx.ByteStream(b"still limited"),
        request=httpx.Request("POST", "https://api.example.test"),
    )
    send = AsyncMock(side_effect=[second])
    monkeypatch.setattr("open_llm_proxy.provider_github_copilot.asyncio.sleep", AsyncMock())

    result = await _send_with_429_retry(
        first, send, model="gpt-5-mini", endpoint="https://api.example.test",
    )
    assert result is second
    send.assert_awaited_once()


@pytest.mark.anyio
async def test_streaming_429_retries_and_raises_custom_error(monkeypatch):
    handler = GithubCopilotLLM()
    monkeypatch.setattr(
        copilot_creds,
        "get_copilot_token",
        AsyncMock(return_value=("mock_tok", "https://api.copilot.com")),
    )
    monkeypatch.setattr(
        handler, "get_endpoint_for_model", AsyncMock(return_value="/chat/completions")
    )
    monkeypatch.setattr(
        "open_llm_proxy.provider_github_copilot.asyncio.sleep", AsyncMock()
    )

    responses = [
        httpx.Response(
            429,
            headers={"retry-after": "0.01", "x-request-id": "req-1"},
            stream=httpx.ByteStream(b"Bearer gho_stream_secret"),
            request=httpx.Request("POST", "https://api.copilot.com/chat/completions"),
        ),
        httpx.Response(
            429,
            headers={"retry-after": "0.01"},
            stream=httpx.ByteStream(b"still limited"),
            request=httpx.Request("POST", "https://api.copilot.com/chat/completions"),
        ),
    ]
    client = MagicMock(spec=httpx.AsyncClient)
    client.build_request = MagicMock()
    client.send = AsyncMock(side_effect=responses)
    monkeypatch.setattr(handler, "_get_client", AsyncMock(return_value=client))

    with pytest.raises(CustomLLMError) as exc_info:
        async for _ in handler.astreaming(
            model="github-copilot/gpt-5-mini",
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass

    assert exc_info.value.status_code == 429
    assert client.send.await_count == 2


def test_429_diagnostic_redacts_before_truncation(caplog):
    response = httpx.Response(
        429,
        text="upstream echo: Bearer gho_test_secret and gho_other_secret",
    )
    with caplog.at_level("WARNING", logger="open_llm_proxy.provider_github_copilot"):
        _log_429_diagnostic(response, "gpt-5-mini", "https://api.example.test")

    message = caplog.records[-1].message
    assert "gho_test_secret" not in message
    assert "gho_other_secret" not in message
    assert "[REDACTED]" in message
    assert message.count("[REDACTED]") == 2


@pytest.mark.parametrize(
    ("text", "secrets"),
    [
        (
            "Authorization: Bearer arbitrary_secret_not_prefixed_by_github_token",
            ["arbitrary_secret_not_prefixed_by_github_token"],
        ),
        ("authorization: bearer SECRET", ["SECRET"]),
        ("Authorization: gho_abc123", ["gho_abc123"]),
        ("Bearer SECRET", ["SECRET"]),
        (
            "Bearer first_secret and gho_second_secret",
            ["first_secret", "gho_second_secret"],
        ),
        ("prefix\nBearer newline_secret", ["newline_secret"]),
        (
            'Bearer semicolon_secret; Bearer comma_secret, Bearer quote_secret"',
            ["semicolon_secret", "comma_secret", "quote_secret"],
        ),
    ],
)
def test_429_diagnostic_redacts_all_secret_forms(text, secrets):
    redacted = _redact_429_diagnostic(text)

    for secret in secrets:
        assert secret not in redacted


@pytest.mark.anyio
async def test_get_endpoint_for_model_dynamic(monkeypatch):
    handler = GithubCopilotLLM()

    # Mock the HTTP GET to /models
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"id": "gpt-5.5", "supported_endpoints": ["/responses"]},
            {"id": "gpt-5-mini", "supported_endpoints": ["/chat/completions"]},
            {"id": "custom-model", "supported_endpoints": ["/responses"]}
        ]
    }

    async def mock_get_client():
        m_client = MagicMock(spec=httpx.AsyncClient)
        m_client.get = AsyncMock(return_value=mock_response)
        return m_client

    monkeypatch.setattr(handler, "_get_client", mock_get_client)
    monkeypatch.setattr(copilot_creds, "get_copilot_token", AsyncMock(return_value=("mock_tok", "https://api.copilot.com")))

    ep = await handler.get_endpoint_for_model("github-copilot/custom-model")
    assert ep == "/responses"

    ep_heuristic = await handler.get_endpoint_for_model("github-copilot/gpt-4o")
    assert ep_heuristic == "/chat/completions"


def test_initiator_header_selection():
    # User message -> initiator user
    assert _initiator_for({"messages": [{"role": "user", "content": "hi"}]}) == "user"

    # Tool message last -> initiator agent
    assert _initiator_for({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "call_123", "content": "done"}
    ]}) == "agent"

    # Assistant message last -> initiator agent
    assert _initiator_for({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"}
    ]}) == "agent"


def test_copilot_vision_request_detection():
    # Regular text -> False
    assert not _has_image_part({"messages": [{"role": "user", "content": "hello"}]})

    # Image payload list content -> True
    assert _has_image_part({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": "foo"}]}
    ]})


def test_tool_call_id_fidelity_responses_translation():
    # Verify tool_calls ids are preserved verbatim during round-trip translation of /responses
    tool_id = "my-special-tool-call-id-99"

    # 1. Inbound OpenAI request with tool call output
    openai_req = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "user", "content": "run tool"},
            {"role": "assistant", "tool_calls": [{"id": tool_id, "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": tool_id, "content": "sunny"}
        ]
    }

    translated_req = copilot_chat_to_responses(openai_req)

    # Assert tool call and output have verbatim tool_id in the copilot input array
    input_items = translated_req["input"]
    assistant_tc_item = next(item for item in input_items if item.get("type") == "function_call")
    tool_output_item = next(item for item in input_items if item.get("type") == "function_call_output")

    assert assistant_tc_item["call_id"] == tool_id
    assert tool_output_item["call_id"] == tool_id

    # 2. Outbound Copilot response translated back to OpenAI
    copilot_resp = {
        "output": [
            {
                "type": "function_call",
                "call_id": tool_id,
                "name": "get_weather",
                "arguments": "{}"
            }
        ],
        "copilot_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }

    openai_resp = copilot_responses_to_chat(
        copilot_resp,
        completion_id="chatcmpl-123",
        model="github-copilot/gpt-5.5"
    )

    tcalls = openai_resp["choices"][0]["message"]["tool_calls"]
    assert len(tcalls) == 1
    assert tcalls[0]["id"] == tool_id


@pytest.mark.anyio
async def test_error_mapping_429_custom_llm_error(monkeypatch):
    handler = GithubCopilotLLM()

    # Mock token exchange
    monkeypatch.setattr(copilot_creds, "get_copilot_token", AsyncMock(return_value=("mock_tok", "https://api.copilot.com")))

    # Mock client to return 429
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 429
    mock_resp.text = "Too Many Requests"
    mock_resp.headers = {"retry-after": "30"}

    async def mock_get_client():
        m_client = MagicMock(spec=httpx.AsyncClient)
        # Mock send to return 429
        m_client.build_request = MagicMock()
        m_client.send = AsyncMock(return_value=mock_resp)
        return m_client

    monkeypatch.setattr(handler, "_get_client", mock_get_client)
    monkeypatch.setattr(handler, "get_endpoint_for_model", AsyncMock(return_value="/chat/completions"))

    with pytest.raises(CustomLLMError) as exc_info:
        await handler.acompletion(
            model="github-copilot/gpt-5-mini",
            messages=[{"role": "user", "content": "hi"}]
        )
    assert exc_info.value.status_code == 429
    assert exc_info.value.headers == {"retry-after": "30"}
    assert (
        exc_info.value.rate_limit_origin_key
        == "github-copilot/gpt-5-mini"
    )


@pytest.mark.anyio
async def test_chat_stream_preserves_tool_calls_finish_reason(monkeypatch):
    handler = GithubCopilotLLM()
    monkeypatch.setattr(
        copilot_creds,
        "get_copilot_token",
        AsyncMock(return_value=("mock_tok", "https://api.copilot.com")),
    )
    monkeypatch.setattr(
        handler, "get_endpoint_for_model", AsyncMock(return_value="/chat/completions")
    )

    class StreamingResponse:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"bash","arguments":"{}"}}]},"finish_reason":null}]}'
            yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            yield 'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}'
            yield "data: [DONE]"

        async def aclose(self):
            pass

    client = MagicMock(spec=httpx.AsyncClient)
    client.build_request = MagicMock()
    client.send = AsyncMock(return_value=StreamingResponse())
    monkeypatch.setattr(handler, "_get_client", AsyncMock(return_value=client))

    chunks = [
        chunk
        async for chunk in handler.astreaming(
            model="github-copilot/claude-opus-4.8",
            messages=[{"role": "user", "content": "run tool"}],
        )
    ]

    assert len(chunks) == 3
    assert chunks[0]["tool_use"]["id"] == "call-1"
    finished = [chunk for chunk in chunks if chunk["is_finished"]]
    assert len(finished) == 1
    assert finished[0]["finish_reason"] == "tool_calls"


def test_copilot_chat_to_responses_role_based_content_parts():
    # 1. assistant message with string content -> output_text
    req_assistant_str = {
        "model": "gpt-5.5",
        "messages": [{"role": "assistant", "content": "hello from assistant"}]
    }
    res = copilot_chat_to_responses(req_assistant_str)
    assert res["input"][0]["content"] == [{"type": "output_text", "text": "hello from assistant"}]

    # 2. user message with string content -> input_text
    req_user_str = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "hello from user"}]
    }
    res = copilot_chat_to_responses(req_user_str)
    assert res["input"][0]["content"] == [{"type": "input_text", "text": "hello from user"}]

    # 3. system message -> input_text
    req_system_str = {
        "model": "gpt-5.5",
        "messages": [{"role": "system", "content": "you are a system"}]
    }
    res = copilot_chat_to_responses(req_system_str)
    assert res["input"][0]["content"] == [{"type": "input_text", "text": "you are a system"}]

    # 4. assistant message with a list of parts -> output_text
    req_assistant_list = {
        "model": "gpt-5.5",
        "messages": [{
            "role": "assistant",
            "content": [{"type": "text", "text": "structured response"}]
        }]
    }
    res = copilot_chat_to_responses(req_assistant_list)
    assert res["input"][0]["content"] == [{"type": "output_text", "text": "structured response"}]


@pytest.mark.anyio
async def test_chat_stream_non_200_error_body_handling(monkeypatch):
    handler = GithubCopilotLLM()
    monkeypatch.setattr(
        copilot_creds,
        "get_copilot_token",
        AsyncMock(return_value=("mock_tok", "https://api.copilot.com")),
    )
    monkeypatch.setattr(
        handler, "get_endpoint_for_model", AsyncMock(return_value="/chat/completions")
    )

    class MockResponse_200_fail:
        status_code = 400
        is_closed = False

        async def aiter_bytes(self, chunk_size=1024):
            yield b"Some error details here"

        async def aclose(self):
            self.is_closed = True

    mock_resp = MockResponse_200_fail()
    client = MagicMock(spec=httpx.AsyncClient)
    client.build_request = MagicMock()
    client.send = AsyncMock(return_value=mock_resp)
    monkeypatch.setattr(handler, "_get_client", AsyncMock(return_value=client))

    # A genuine HTTP 400 must surface as a non-retriable BadRequestError, not a
    # CustomLLMError (which LiteLLM maps to a retriable APIConnectionError for
    # custom providers, causing an infinite fallback retry loop).
    from litellm.exceptions import BadRequestError

    with pytest.raises(BadRequestError) as exc_info:
        async for _ in handler.astreaming(
            model="github-copilot/claude-opus-4.8",
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass

    assert "Some error details here" in exc_info.value.message
    assert mock_resp.is_closed


@pytest.mark.anyio
async def test_chat_stream_non_200_empty_body_handling(monkeypatch):
    handler = GithubCopilotLLM()
    monkeypatch.setattr(
        copilot_creds,
        "get_copilot_token",
        AsyncMock(return_value=("mock_tok", "https://api.copilot.com")),
    )
    monkeypatch.setattr(
        handler, "get_endpoint_for_model", AsyncMock(return_value="/chat/completions")
    )

    class MockResponse_empty:
        status_code = 500
        is_closed = False

        async def aiter_bytes(self, chunk_size=1024):
            # empty body
            if False:
                yield b""

        async def aclose(self):
            self.is_closed = True

    mock_resp = MockResponse_empty()
    client = MagicMock(spec=httpx.AsyncClient)
    client.build_request = MagicMock()
    client.send = AsyncMock(return_value=mock_resp)
    monkeypatch.setattr(handler, "_get_client", AsyncMock(return_value=client))

    with pytest.raises(CustomLLMError) as exc_info:
        async for _ in handler.astreaming(
            model="github-copilot/claude-opus-4.8",
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass

    assert exc_info.value.message == "HTTP 500"
    assert mock_resp.is_closed


@pytest.mark.anyio
async def test_chat_stream_non_200_oversized_truncation(monkeypatch):
    handler = GithubCopilotLLM()
    monkeypatch.setattr(
        copilot_creds,
        "get_copilot_token",
        AsyncMock(return_value=("mock_tok", "https://api.copilot.com")),
    )
    monkeypatch.setattr(
        handler, "get_endpoint_for_model", AsyncMock(return_value="/chat/completions")
    )

    class MockResponse_large:
        status_code = 403
        is_closed = False

        async def aiter_bytes(self, chunk_size=1024):
            # Yield 5 chunks of 1024 bytes
            for _ in range(5):
                yield b"A" * 1024

        async def aclose(self):
            self.is_closed = True

    mock_resp = MockResponse_large()
    client = MagicMock(spec=httpx.AsyncClient)
    client.build_request = MagicMock()
    client.send = AsyncMock(return_value=mock_resp)
    monkeypatch.setattr(handler, "_get_client", AsyncMock(return_value=client))

    with pytest.raises(CustomLLMError) as exc_info:
        async for _ in handler.astreaming(
            model="github-copilot/claude-opus-4.8",
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass

    # Exact length: 4096 (A's) + "... [truncated]" length (15) = 4111
    assert len(exc_info.value.message) == 4111
    assert exc_info.value.message.endswith("... [truncated]")
    assert mock_resp.is_closed


@pytest.mark.anyio
async def test_chat_stream_non_200_invalid_utf8_replacement(monkeypatch):
    handler = GithubCopilotLLM()
    monkeypatch.setattr(
        copilot_creds,
        "get_copilot_token",
        AsyncMock(return_value=("mock_tok", "https://api.copilot.com")),
    )
    monkeypatch.setattr(
        handler, "get_endpoint_for_model", AsyncMock(return_value="/chat/completions")
    )

    class MockResponse_invalid_utf8:
        status_code = 502
        is_closed = False

        async def aiter_bytes(self, chunk_size=1024):
            yield b"Error: \xff\xff invalid bytes"

        async def aclose(self):
            self.is_closed = True

    mock_resp = MockResponse_invalid_utf8()
    client = MagicMock(spec=httpx.AsyncClient)
    client.build_request = MagicMock()
    client.send = AsyncMock(return_value=mock_resp)
    monkeypatch.setattr(handler, "_get_client", AsyncMock(return_value=client))

    with pytest.raises(CustomLLMError) as exc_info:
        async for _ in handler.astreaming(
            model="github-copilot/claude-opus-4.8",
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass

    assert "Error: \ufffd\ufffd invalid bytes" in exc_info.value.message
    assert mock_resp.is_closed


@pytest.mark.anyio
async def test_chat_stream_non_200_exact_4096_boundary(monkeypatch):
    handler = GithubCopilotLLM()
    monkeypatch.setattr(
        copilot_creds,
        "get_copilot_token",
        AsyncMock(return_value=("mock_tok", "https://api.copilot.com")),
    )
    monkeypatch.setattr(
        handler, "get_endpoint_for_model", AsyncMock(return_value="/chat/completions")
    )

    class MockResponse_exact_4096:
        status_code = 401
        is_closed = False

        async def aiter_bytes(self, chunk_size=1024):
            # Yield exactly 4 chunks of 1024 bytes
            for _ in range(4):
                yield b"B" * 1024

        async def aclose(self):
            self.is_closed = True

    mock_resp = MockResponse_exact_4096()
    client = MagicMock(spec=httpx.AsyncClient)
    client.build_request = MagicMock()
    client.send = AsyncMock(return_value=mock_resp)
    monkeypatch.setattr(handler, "_get_client", AsyncMock(return_value=client))

    with pytest.raises(CustomLLMError) as exc_info:
        async for _ in handler.astreaming(
            model="github-copilot/claude-opus-4.8",
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass

    assert exc_info.value.status_code == 401
    assert exc_info.value.message == "B" * 4096
    assert mock_resp.is_closed


# ── LIVE SHIELDED COMPLETION TESTS ───────────────────────────────────────────


@pytest.mark.skipif(not has_copilot, reason="No GitHub Copilot credentials resolved on this machine")
@pytest.mark.anyio
async def test_live_copilot_gpt_5_5_responses_path():
    res = await copilot_handler.acompletion(
        model="github-copilot/gpt-5.5",
        messages=[{"role": "user", "content": "reply with the single word pong"}],
        max_tokens=64
    )
    assert isinstance(res, ModelResponse)
    content = res.choices[0].message.content
    assert content is not None
    assert "pong" in content.lower()


# ── TOOL SCHEMA NORMALIZATION (Copilot boundary defense-in-depth) ─────────────


def _obj_tool(params: dict) -> dict:
    return {"type": "function", "function": {"name": "t", "parameters": params}}


def test_normalize_adds_properties_to_bare_object():
    # A no-argument tool emitted as just {"type": "object"} — Copilot 400s on this.
    tools = [_obj_tool({"type": "object"})]
    out = _normalize_tools_for_copilot(tools)
    assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_normalize_adds_properties_when_only_additionalproperties():
    # The exact shape the Gemini transform can leave behind.
    tools = [_obj_tool({"type": "object", "additionalProperties": False})]
    out = _normalize_tools_for_copilot(tools)
    p = out[0]["function"]["parameters"]
    assert p["properties"] == {}
    assert p["additionalProperties"] is False


def test_normalize_preserves_existing_properties():
    params = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    out = _normalize_tools_for_copilot([_obj_tool(params)])
    assert out[0]["function"]["parameters"] == params


def test_normalize_recurses_into_nested_objects_arrays_and_defs():
    params = {
        "type": "object",
        "properties": {
            "nested": {"type": "object"},  # missing properties, nested
            "arr": {"type": "array", "items": {"type": "object"}},  # under items
        },
        "$defs": {"D": {"type": "object"}},  # under $defs
    }
    out = _normalize_tools_for_copilot([_obj_tool(params)])
    p = out[0]["function"]["parameters"]
    assert p["properties"]["nested"]["properties"] == {}
    assert p["properties"]["arr"]["items"]["properties"] == {}
    assert p["$defs"]["D"]["properties"] == {}


def test_normalize_does_not_mutate_caller_object():
    # The router shares one tools list across every fallback deployment; mutating
    # it here would corrupt the request for other providers.
    original = _obj_tool({"type": "object", "additionalProperties": False})
    tools = [original]
    _normalize_tools_for_copilot(tools)
    assert "properties" not in original["function"]["parameters"]


def test_normalize_leaves_non_object_schemas_untouched():
    params = {"type": "string", "enum": ["a", "b"]}
    out = _normalize_tools_for_copilot([_obj_tool(params)])
    assert out[0]["function"]["parameters"] == {"type": "string", "enum": ["a", "b"]}


def test_ensure_object_schemas_handles_combinators():
    node = {"anyOf": [{"type": "object"}, {"type": "string"}]}
    _ensure_object_schemas_have_properties(node)
    assert node["anyOf"][0]["properties"] == {}
    assert node["anyOf"][1] == {"type": "string"}


@pytest.mark.skipif(not has_copilot, reason="No GitHub Copilot credentials resolved on this machine")
@pytest.mark.anyio
async def test_live_copilot_gpt_5_mini_chat_path():
    res = await copilot_handler.acompletion(
        model="github-copilot/gpt-5-mini",
        messages=[{"role": "user", "content": "reply with the single word pong"}],
        max_tokens=64
    )
    assert isinstance(res, ModelResponse)
    content = res.choices[0].message.content
    assert content is not None
    assert "pong" in content.lower()


class TestResponsesImagePartTranslation:
    """The /responses endpoint rejects chat ``image_url`` parts outright.

    Verified live against api.enterprise.githubcopilot.com: sending
    ``{"type": "image_url", ...}`` to /responses returns HTTP 400
    ``Invalid value: 'image_url'. Supported values are: 'input_text',
    'input_image', ...`` while the translated ``input_image`` shape returns 200.
    """

    def test_image_url_object_becomes_input_image_with_string_url(self):
        url = "data:image/png;base64,aGVsbG8="
        req = {
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what colour?"},
                {"type": "image_url", "image_url": {"url": url}},
            ]}],
        }

        content = copilot_chat_to_responses(req)["input"][0]["content"]

        assert content[0] == {"type": "input_text", "text": "what colour?"}
        assert content[1] == {"type": "input_image", "image_url": url}

    def test_bare_string_image_url_is_also_translated(self):
        req = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": "https://example.com/a.png"},
        ]}]}

        part = copilot_chat_to_responses(req)["input"][0]["content"][0]

        assert part == {"type": "input_image", "image_url": "https://example.com/a.png"}

    def test_detail_hint_is_preserved(self):
        req = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://x/a.png", "detail": "high"}},
        ]}]}

        part = copilot_chat_to_responses(req)["input"][0]["content"][0]

        assert part["type"] == "input_image" and part["detail"] == "high"

    def test_no_image_url_type_survives_translation(self):
        req = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
        ]}]}

        for item in copilot_chat_to_responses(req)["input"]:
            for part in item["content"]:
                assert part.get("type") != "image_url"

    def test_unrecoverable_image_part_passes_through_untouched(self):
        part = {"type": "image_url", "image_url": {}}
        req = {"messages": [{"role": "user", "content": [part]}]}

        assert copilot_chat_to_responses(req)["input"][0]["content"][0] is part

    def test_assistant_and_other_part_types_are_unaffected(self):
        req = {"messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "prior"}]},
            {"role": "user", "content": [{"type": "input_file", "filename": "a.pdf"}]},
        ]}

        result = copilot_chat_to_responses(req)["input"]

        assert result[0]["content"][0]["type"] == "output_text"
        assert result[1]["content"][0] == {"type": "input_file", "filename": "a.pdf"}
