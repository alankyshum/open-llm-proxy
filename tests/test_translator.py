from __future__ import annotations

import pytest

from open_llm_proxy import translator
from open_llm_proxy.errors import TranslationError


def test_billing_block_first():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    payload = translator.build_anthropic_payload(
        model="claude-sonnet-5",
        openai_messages=messages,
        openai_tools=None,
        thinking_level="none",
        max_tokens=1000,
        temperature=0.7,
        stream=False,
    )
    assert "system" in payload
    assert isinstance(payload["system"], list)
    assert len(payload["system"]) >= 1
    first_block = payload["system"][0]
    assert first_block["type"] == "text"
    assert "cc_entrypoint=sdk-cli;" in first_block["text"]

    # If system message was present, it should be the second block
    assert len(payload["system"]) == 2
    assert payload["system"][1]["type"] == "text"
    assert payload["system"][1]["text"] == "You are a helpful assistant."


def test_thinking_level_budget_and_temperature():
    messages = [{"role": "user", "content": "Hello!"}]
    # Thinking high
    payload_high = translator.build_anthropic_payload(
        model="claude-sonnet-5",
        openai_messages=messages,
        openai_tools=None,
        thinking_level="high",
        max_tokens=1000,
        temperature=0.7,
        stream=False,
    )
    assert "thinking" in payload_high
    assert payload_high["thinking"]["type"] == "enabled"
    assert payload_high["thinking"]["budget_tokens"] == 16000
    assert "temperature" not in payload_high

    # Thinking none
    payload_none = translator.build_anthropic_payload(
        model="claude-sonnet-5",
        openai_messages=messages,
        openai_tools=None,
        thinking_level="none",
        max_tokens=1000,
        temperature=0.7,
        stream=False,
    )
    assert "thinking" not in payload_none
    assert payload_none["temperature"] == 0.7


def test_model_catalog_entries_present():
    models = translator.build_models_list()
    ids = {m["id"] for m in models}
    
    # Assert standard canonical models are present
    assert "claude-sonnet-5" in ids
    assert "claude-opus-4-8" in ids
    assert "claude-haiku-4-5" in ids

    # Assert thinking variants exist for each model
    assert "claude-sonnet-5:high" in ids
    assert "claude-sonnet-5:none" in ids


def test_tool_result_without_tool_use_id_raises_translation_error():
    messages = [
        {"role": "user", "content": "run command"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]},
        {"role": "tool", "content": "file1.txt"}  # missing tool_call_id!
    ]
    with pytest.raises(TranslationError) as exc_info:
        translator.openai_messages_to_anthropic(messages)
    assert "missing tool_call_id" in str(exc_info.value)


def test_basic_round_trip():
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "I am Claude."},
    ]
    system_blocks, anth_msgs = translator.openai_messages_to_anthropic(messages)
    
    assert len(system_blocks) == 1
    assert system_blocks[0]["text"] == "System prompt"
    
    assert len(anth_msgs) == 2
    assert anth_msgs[0]["role"] == "user"
    assert anth_msgs[0]["content"] == [{"type": "text", "text": "Who are you?"}]
    assert anth_msgs[1]["role"] == "assistant"
    assert anth_msgs[1]["content"] == [{"type": "text", "text": "I am Claude."}]


def test_claude_opus_dotted_normalization():
    base, level = translator.parse_model("claude-opus-4.8")
    assert base == "claude-opus-4-8"
    assert level == "xhigh"

    base_v, level_v = translator.parse_model("claude-opus-4.8:none")
    assert base_v == "claude-opus-4-8"
    assert level_v == "none"

    assert translator._normalize_model_id("claude-opus-4.8") == "claude-opus-4-8"
