from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from litellm import Router
from litellm.llms.custom_llm import CustomLLMError
from litellm.utils import ModelResponse

from open_llm_proxy import anthropic_client, creds
from open_llm_proxy.errors import RateLimitError
from open_llm_proxy.provider_claude_cli import claude_cli_handler


def resolve_key():
    try:
        return creds.get_api_key()
    except Exception:
        return None


has_creds = resolve_key() is not None


# 1. Offline unit tests with mocked anthropic_client

async def mock_stream_messages_success(*args, **kwargs):
    yield "message_start", {"message": {"usage": {"input_tokens": 10}}}
    yield "content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}
    yield "content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "hello"}}
    yield "content_block_stop", {"index": 0}
    yield "message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}


@pytest.mark.anyio
async def test_claude_cli_streaming_success():
    with patch.object(anthropic_client, "stream_messages", side_effect=mock_stream_messages_success):
        chunks = []
        async for chunk in claude_cli_handler.astreaming(
            model="claude-cli/claude-sonnet-5",
            messages=[{"role": "user", "content": "hello"}],
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Check text delivery
        texts = [ch["text"] for ch in chunks if ch["text"]]
        assert "".join(texts) == "hello"

        # Check last chunk
        last_chunk = chunks[-1]
        assert last_chunk["is_finished"] is True
        assert last_chunk["finish_reason"] == "stop"
        assert last_chunk["usage"] is not None
        assert last_chunk["usage"]["prompt_tokens"] == 10
        assert last_chunk["usage"]["completion_tokens"] == 5
        assert last_chunk["usage"]["total_tokens"] == 15


@pytest.mark.anyio
async def test_claude_cli_rate_limit_error_mapping():
    async def mock_stream_messages_429(*args, **kwargs):
        raise RateLimitError("Mocked Rate Limit", retry_after=5)
        # Yield statement so python treats this as a generator
        yield "ping", {}

    with patch.object(anthropic_client, "stream_messages", side_effect=mock_stream_messages_429):
        with pytest.raises(CustomLLMError) as exc_info:
            async for _ in claude_cli_handler.astreaming(
                model="claude-cli/claude-sonnet-5",
                messages=[{"role": "user", "content": "hello"}],
            ):
                pass
        assert exc_info.value.status_code == 429
        assert "Mocked Rate Limit" in exc_info.value.message


@pytest.mark.anyio
async def test_claude_cli_completion_success():
    mock_raw_response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello back"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 8},
    }

    with patch.object(anthropic_client, "send_messages", return_value=mock_raw_response):
        res = await claude_cli_handler.acompletion(
            model="claude-cli/claude-sonnet-5",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert isinstance(res, ModelResponse)
        assert res.choices[0].message.content == "hello back"
        assert res.choices[0].finish_reason == "stop"
        assert res.usage.prompt_tokens == 12
        assert res.usage.completion_tokens == 8
        assert res.usage.total_tokens == 20


# 2. Offline 429 -> fallback Router test (Encoding 1)

class MockFallbackLLM(litellm.CustomLLM):
    def completion(self, *args, **kwargs):
        return ModelResponse(
            choices=[{
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "fallback-ok"}
            }]
        )


@pytest.mark.anyio
async def test_router_429_fallback():
    from litellm.utils import custom_llm_setup
    # Register our mock fallback provider
    if not any(item.get("provider") == "mock-fallback" for item in litellm.custom_provider_map):
        litellm.custom_provider_map.append({
            "provider": "mock-fallback",
            "custom_handler": MockFallbackLLM()
        })
    custom_llm_setup()

    bracket_alias = "open-llm-proxy/[claude-cli/claude-sonnet-5,mock-fallback/model]"
    model_list = [
        {
            "model_name": bracket_alias,
            "litellm_params": {
                "model": "claude-cli/claude-sonnet-5",
            }
        },
        {
            "model_name": "fallback-model",
            "litellm_params": {
                "model": "mock-fallback/model",
            }
        }
    ]

    router = Router(
        model_list=model_list,
        fallbacks=[{bracket_alias: ["fallback-model"]}]
    )

    # Mock send_messages to raise RateLimitError
    with patch.object(anthropic_client, "send_messages", side_effect=RateLimitError("Mocked Rate Limit", retry_after=5)):
        res = router.completion(
            model=bracket_alias,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert res.choices[0].message.content == "fallback-ok"


# 3. Live registration tests (skippable)

@pytest.mark.skipif(not has_creds, reason="No Claude Code/anthropic creds found in environment or Keychain")
def test_live_registration_completion():
    res = litellm.completion(
        model="claude-cli/claude-sonnet-5",
        messages=[{"role": "user", "content": "reply with the single word pong"}],
        max_tokens=64,
    )
    text = res.choices[0].message.content.strip().lower()
    assert "pong" in text


@pytest.mark.skipif(not has_creds, reason="No Claude Code/anthropic creds found in environment or Keychain")
def test_live_registration_streaming():
    res = litellm.completion(
        model="claude-cli/claude-sonnet-5",
        messages=[{"role": "user", "content": "reply with the single word pong"}],
        max_tokens=64,
        stream=True,
    )
    chunks = list(res)
    assert len(chunks) > 0
    texts = []
    for chunk in chunks:
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            if "content" in delta and delta["content"]:
                texts.append(delta["content"])
    full_text = "".join(texts).strip().lower()
    assert "pong" in full_text
