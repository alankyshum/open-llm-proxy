from __future__ import annotations

import httpx
import pytest
from unittest.mock import MagicMock

from open_llm_proxy import connectivity

def test_check_provider_openrouter(monkeypatch):
    monkeypatch.setattr("open_llm_proxy.openrouter_creds.get_persisted_api_key", lambda: "fake-key")

    def mock_send(request: httpx.Request):
        assert request.url.host == "openrouter.ai"
        assert request.headers["authorization"] == "Bearer fake-key"
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(mock_send)
    monkeypatch.setattr("httpx.Client._transport_for_url", lambda self, url: transport)

    ok, status = connectivity.check_provider("openrouter")
    assert ok is True
    assert status == "Ready"

def test_check_provider_opencode(monkeypatch):
    monkeypatch.setattr("open_llm_proxy.opencode_creds.get_opencode_api_key", lambda: "fake-code-key")

    def mock_send(request: httpx.Request):
        assert request.url.host == "opencode.ai"
        assert request.headers["authorization"] == "Bearer fake-code-key"
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(mock_send)
    monkeypatch.setattr("httpx.Client._transport_for_url", lambda self, url: transport)

    ok, status = connectivity.check_provider("opencode")
    assert ok is True
    assert status == "Ready"

def test_check_provider_claude_cli(monkeypatch):
    monkeypatch.setattr("open_llm_proxy.creds.get_api_key", lambda: "fake-claude-key")

    def mock_send(request: httpx.Request):
        assert request.url.host == "api.anthropic.com"
        assert request.headers["x-api-key"] == "fake-claude-key"
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(mock_send)
    monkeypatch.setattr("httpx.Client._transport_for_url", lambda self, url: transport)

    ok, status = connectivity.check_provider("claude-cli")
    assert ok is True
    assert status == "Ready"

def test_check_provider_copilot(monkeypatch):
    async def mock_get_token():
        return "fake-copilot-token", "https://api.githubcopilot.com"

    monkeypatch.setattr("open_llm_proxy.copilot_creds.get_copilot_token", mock_get_token)

    def mock_send(request: httpx.Request):
        assert request.url.host == "api.githubcopilot.com"
        assert request.headers["authorization"] == "Bearer fake-copilot-token"
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(mock_send)
    monkeypatch.setattr("httpx.Client._transport_for_url", lambda self, url: transport)

    ok, status = connectivity.check_provider("github-copilot")
    assert ok is True
    assert status == "Ready"

@pytest.mark.parametrize(
    "status_code, expected_ok, expected_status",
    [
        (401, False, "Authentication Failed"),
        (403, False, "Authentication Failed"),
        (429, False, "Rate Limited"),
        (400, False, "Client Error"),
        (500, False, "Server Error"),
        (503, False, "Server Error"),
    ]
)
def test_check_provider_error_statuses(monkeypatch, status_code, expected_ok, expected_status):
    monkeypatch.setattr("open_llm_proxy.openrouter_creds.get_persisted_api_key", lambda: "fake-key")

    def mock_send(request: httpx.Request):
        return httpx.Response(status_code, text="sensitive error details or secrets in response body")

    transport = httpx.MockTransport(mock_send)
    monkeypatch.setattr("httpx.Client._transport_for_url", lambda self, url: transport)

    ok, status = connectivity.check_provider("openrouter")
    assert ok is expected_ok
    assert status == expected_status

def test_check_provider_timeout(monkeypatch):
    monkeypatch.setattr("open_llm_proxy.openrouter_creds.get_persisted_api_key", lambda: "fake-key")

    def mock_send(request: httpx.Request):
        raise httpx.TimeoutException("mock timeout")

    transport = httpx.MockTransport(mock_send)
    monkeypatch.setattr("httpx.Client._transport_for_url", lambda self, url: transport)

    ok, status = connectivity.check_provider("openrouter")
    assert ok is False
    assert status == "Timeout"

def test_check_provider_connection_failed(monkeypatch):
    monkeypatch.setattr("open_llm_proxy.openrouter_creds.get_persisted_api_key", lambda: "fake-key")

    def mock_send(request: httpx.Request):
        raise httpx.ConnectError("mock connection failed")

    transport = httpx.MockTransport(mock_send)
    monkeypatch.setattr("httpx.Client._transport_for_url", lambda self, url: transport)

    ok, status = connectivity.check_provider("openrouter")
    assert ok is False
    assert status == "Connection Failed"
