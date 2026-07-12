from __future__ import annotations

import httpx
import pytest
from pathlib import Path

from open_llm_proxy import nvidia_creds, env_creds


@pytest.fixture
def cfg_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set OLP_CONFIG_DIR to a temp directory."""
    d = tmp_path / "olp_config"
    d.mkdir()
    monkeypatch.setenv("OLP_CONFIG_DIR", str(d))
    return d


class TestNvidiaCreds:
    def test_save_and_get(self, cfg_dir):
        nvidia_creds.save_api_key("nv-api-key-123")
        assert nvidia_creds.get_api_key() == "nv-api-key-123"

    def test_get_raises_when_absent(self, cfg_dir):
        with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
            nvidia_creds.get_api_key()

    def test_round_trip_via_env_creds(self, cfg_dir):
        nvidia_creds.save_api_key("nv-secret")
        assert env_creds.get_env_key("NVIDIA_API_KEY") == "nv-secret"

    def test_get_from_env_var(self, cfg_dir, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nv-from-env")
        assert nvidia_creds.get_api_key() == "nv-from-env"

    def test_reads_from_existing_env_file(self, cfg_dir):
        env_file = cfg_dir / "env"
        env_file.write_text("NVIDIA_API_KEY=nv-file-key\n", encoding="utf-8")
        assert nvidia_creds.get_api_key() == "nv-file-key"


class TestNvidiaConnectivity:
    """Verify the nvidia branch in connectivity.check_provider builds correct
    URL and headers."""

    def test_probe_url_and_headers(self, cfg_dir, monkeypatch):
        nvidia_creds.save_api_key("nv-probe-key")

        captured_url = None
        captured_headers = None

        def mock_get(url, *, headers=None, **kw):
            nonlocal captured_url, captured_headers
            captured_url = url
            captured_headers = headers
            return httpx.Response(200, json={})

        def mock_send(request: httpx.Request):
            nonlocal captured_url, captured_headers
            return httpx.Response(200, json={})

        transport = httpx.MockTransport(mock_send)
        monkeypatch.setattr("httpx.Client._transport_for_url", lambda self, url: transport)

        from open_llm_proxy import connectivity

        ok, status = connectivity.check_provider("nvidia")
        assert ok is True
        assert status == "Ready"

    def test_auth_failure_401(self, cfg_dir, monkeypatch):
        nvidia_creds.save_api_key("bad-key")

        def mock_send(request: httpx.Request):
            assert "Bearer bad-key" in request.headers.get("authorization", "")
            assert request.url.host == "integrate.api.nvidia.com"
            return httpx.Response(401, json={})

        transport = httpx.MockTransport(mock_send)
        monkeypatch.setattr("httpx.Client._transport_for_url", lambda self, url: transport)

        from open_llm_proxy import connectivity

        ok, status = connectivity.check_provider("nvidia")
        assert ok is False
        assert status == "Authentication Failed"

    def test_missing_credentials(self, cfg_dir):
        from open_llm_proxy import connectivity

        ok, status = connectivity.check_provider("nvidia")
        assert ok is False
        assert status == "Missing Credentials"
