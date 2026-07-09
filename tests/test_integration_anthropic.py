from __future__ import annotations

import pytest

from open_llm_proxy import creds, anthropic_client, translator
from open_llm_proxy.errors import RateLimitError


def resolve_key():
    try:
        return creds.get_api_key()
    except Exception:
        return None


has_creds = resolve_key() is not None


@pytest.mark.skipif(not has_creds, reason="No Claude Code/anthropic creds found in environment or Keychain")
@pytest.mark.anyio
async def test_sonnet_stream_unlock():
    # Reset global client to avoid Event Loop Closed error from prior tests
    anthropic_client._client = None

    # 1. Resolve key
    key = creds.get_api_key()
    assert key is not None

    # 2. Build payload
    messages = [{"role": "user", "content": "reply with the single word: pong"}]
    payload = translator.build_anthropic_payload(
        model="claude-sonnet-5",
        openai_messages=messages,
        openai_tools=None,
        thinking_level="none",
        max_tokens=64,
        temperature=None,
        stream=True,
    )

    # Confirm first system block contains cc_entrypoint=sdk-cli;
    assert "system" in payload
    assert isinstance(payload["system"], list)
    assert len(payload["system"]) >= 1
    first_block = payload["system"][0]
    assert first_block["type"] == "text"
    assert "cc_entrypoint=sdk-cli;" in first_block["text"], "cc_entrypoint=sdk-cli; must be in the first system block"

    # 3. Stream messages
    assistant_text = []
    try:
        async for event_name, event_data in anthropic_client.stream_messages(payload):
            if event_name == "content_block_delta":
                delta = event_data.get("delta") or {}
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    assistant_text.append(text)
    except RateLimitError as e:
        pytest.fail(f"Tier-0 unlock failed: 429 RateLimitError raised: {e}")
    except Exception as e:
        pytest.fail(f"Streaming failed with exception: {e}")
    finally:
        anthropic_client._client = None

    full_text = "".join(assistant_text).strip()
    assert len(full_text) > 0, "Assistant response was empty"
    print(f"\n[INTEGRATION TEST] Assistant text returned: {full_text}")
