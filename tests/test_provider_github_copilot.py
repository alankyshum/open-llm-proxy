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
    copilot_chat_to_responses,
    copilot_responses_to_chat,
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
