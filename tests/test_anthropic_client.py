from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from open_llm_proxy import anthropic_client, creds
from open_llm_proxy.errors import RateLimitError


class MockResponse:
    def __init__(self, status_code, content=b"", headers=None, lines=None):
        self.status_code = status_code
        self._content = content
        self.headers = headers or {}
        self._lines = lines or []

    async def aread(self):
        return self._content

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class AsyncContextManagerMock:
    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture(autouse=True)
def cleanup():
    creds.clear_cache()
    creds.reset_oauth_state()


def test_headers_selection_non_oauth():
    headers = anthropic_client._headers("sk-ant-api03-1234")
    assert headers["x-api-key"] == "sk-ant-api03-1234"
    assert "Authorization" not in headers
    assert "anthropic-beta" not in headers


def test_headers_selection_oauth():
    headers = anthropic_client._headers("sk-ant-oat01-abcde")
    assert headers["Authorization"] == "Bearer sk-ant-oat01-abcde"
    assert headers["anthropic-beta"] == "oauth-2025-04-20"
    assert "x-api-key" not in headers


@pytest.mark.anyio
async def test_stream_messages_success():
    payload = {"model": "claude-sonnet-5", "messages": []}
    lines = [
        "event: message_start",
        'data: {"type": "message_start"}',
        "",
        "event: content_block_delta",
        'data: {"delta": {"text": "hello"}}',
        "",
    ]
    mock_resp = MockResponse(200, lines=lines)

    with patch.object(creds, "get_api_key", return_value="sk-ant-api03-test"), \
         patch.object(httpx.AsyncClient, "stream", return_value=AsyncContextManagerMock(mock_resp)):
        events = []
        async for ev, data in anthropic_client.stream_messages(payload):
            events.append((ev, data))

        assert len(events) == 2
        assert events[0][0] == "message_start"
        assert events[1][0] == "content_block_delta"
        assert events[1][1]["delta"]["text"] == "hello"


@pytest.mark.anyio
async def test_stream_messages_401_retry():
    payload = {"model": "claude-sonnet-5", "messages": []}
    mock_401_resp = MockResponse(401, content=b"Unauthorized")
    mock_200_resp = MockResponse(200, lines=["event: message_start", 'data: {"type": "message_start"}', ""])

    call_count = 0

    def mock_stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AsyncContextManagerMock(mock_401_resp)
        return AsyncContextManagerMock(mock_200_resp)

    with patch.object(creds, "get_api_key", return_value="sk-ant-api03-old") as mock_get_key, \
         patch.object(creds, "refresh_anthropic_oauth") as mock_refresh, \
         patch.object(httpx.AsyncClient, "stream", side_effect=mock_stream):
        
        events = []
        async for ev, data in anthropic_client.stream_messages(payload):
            events.append((ev, data))

        assert len(events) == 1
        assert events[0][0] == "message_start"
        assert mock_refresh.call_count == 1
        mock_refresh.assert_called_with(stale_token="sk-ant-api03-old")


@pytest.mark.anyio
async def test_stream_messages_429_rate_limit():
    payload = {"model": "claude-sonnet-5", "messages": []}
    mock_429_resp = MockResponse(429, content=b"Rate Limit Exceeded", headers={"retry-after": "15"})

    with patch.object(creds, "get_api_key", return_value="sk-ant-api03-test"), \
         patch.object(httpx.AsyncClient, "stream", return_value=AsyncContextManagerMock(mock_429_resp)):
        
        with pytest.raises(RateLimitError) as exc_info:
            async for _, _ in anthropic_client.stream_messages(payload):
                pass

        assert exc_info.value.retry_after == 15.0
        assert "Rate Limit Exceeded" in str(exc_info.value)


@pytest.mark.anyio
async def test_integration_call_skippable():
    # Attempt a real call if we have credentials, else skip.
    try:
        key = creds.get_api_key()
    except Exception:
        pytest.skip("No real credentials present for integration testing.")

    if not key.startswith("sk-"):
        pytest.skip("Credentials do not look like a standard API key. Skipping real call.")

    payload = {
        "model": "claude-haiku-4-5",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hello"}],
    }

    try:
        resp = await anthropic_client.send_messages(payload)
        assert "content" in resp
    except Exception as e:
        pytest.fail(f"Real API call failed: {e}")
