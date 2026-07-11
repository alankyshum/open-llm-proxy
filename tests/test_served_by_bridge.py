import uuid
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from open_llm_proxy.attribution import (
    AttributionStore,
    attribution_id_from_data,
    get_attribution_token,
    global_attribution_store,
)
from open_llm_proxy.callbacks import ServedByCallback
from open_llm_proxy.server_launcher import register_attribution_endpoint

def test_attribution_store_basic():
    store = AttributionStore(capacity=3, ttl=1.0)
    
    # Check invalid UUID does not crash and is ignored
    store.set("not-a-uuid", "model-a")
    assert store.get("not-a-uuid") is None

    uid1 = str(uuid.uuid4())
    uid2 = str(uuid.uuid4())
    uid3 = str(uuid.uuid4())
    uid4 = str(uuid.uuid4())

    store.set(uid1, "model-1")
    store.set(uid2, "model-2")
    store.set(uid3, "model-3")
    
    assert store.get(uid1) == "model-1"
    assert store.get(uid2) == "model-2"
    assert store.get(uid3) == "model-3"

    # Capacity eviction (FIFO oldest is uid1)
    store.set(uid4, "model-4")
    assert store.get(uid1) is None
    assert store.get(uid4) == "model-4"

def test_attribution_store_ttl():
    fake_time = 100.0
    def custom_clock():
        return fake_time

    store = AttributionStore(capacity=10, ttl=5.0, clock=custom_clock)
    uid = str(uuid.uuid4())
    
    store.set(uid, "model-a")
    assert store.get(uid) == "model-a"

    # Move time forward inside TTL
    fake_time = 104.0
    assert store.get(uid) == "model-a"

    # Move past TTL
    fake_time = 106.0
    assert store.get(uid) is None


def test_attribution_store_announces_only_winner_changes():
    store = AttributionStore()
    uid = str(uuid.uuid4())

    assert store.announce_if_changed(uid, "model-a") is True
    assert store.announce_if_changed(uid, "model-a") is False

    # Response-header tracking updates latest winner without consuming the
    # changed-winner announcement used by response-body attribution.
    store.set(uid, "model-b")
    assert store.get(uid) == "model-b"
    assert store.announce_if_changed(uid, "model-b") is True
    assert store.announce_if_changed(uid, "model-b") is False


def test_attribution_id_from_litellm_request_metadata():
    uid = str(uuid.uuid4())
    data = {
        "litellm_params": {
            "proxy_server_request": {
                "headers": {"X-Open-LLM-Proxy-Attribution-ID": uid}
            }
        }
    }

    assert attribution_id_from_data(data) == uid
    assert attribution_id_from_data({}) is None


def test_attribution_token_env_precedes_file(monkeypatch, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("file-token\n")
    monkeypatch.setenv("OPEN_LLM_PROXY_ATTRIBUTION_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("OPEN_LLM_PROXY_ATTRIBUTION_TOKEN", "env-token")
    assert get_attribution_token() == "env-token"


def test_attribution_token_file_fallback(monkeypatch, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("file-token\n")
    token_file.chmod(0o600)
    monkeypatch.delenv("OPEN_LLM_PROXY_ATTRIBUTION_TOKEN", raising=False)
    monkeypatch.setenv("OPEN_LLM_PROXY_ATTRIBUTION_TOKEN_FILE", str(token_file))
    assert get_attribution_token() == "file-token"


def test_attribution_token_rejects_insecure_file(monkeypatch, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("file-token\n")
    token_file.chmod(0o644)
    monkeypatch.delenv("OPEN_LLM_PROXY_ATTRIBUTION_TOKEN", raising=False)
    monkeypatch.setenv("OPEN_LLM_PROXY_ATTRIBUTION_TOKEN_FILE", str(token_file))
    assert get_attribution_token() is None

@pytest.mark.asyncio
async def test_served_by_callback_extract_attribution_id():
    callback = ServedByCallback()
    uid = str(uuid.uuid4())
    
    data = {"deployment": {"model_info": {"rate_limit_key": "openai/gpt-4"}}}
    req_headers = {"X-Open-LLM-Proxy-Attribution-ID": uid}
    
    # Clear the global store first
    global_attribution_store.clear()
    
    from types import SimpleNamespace
    class MockResponse:
        def __init__(self):
            self.choices = [SimpleNamespace(message=SimpleNamespace(content="Hi", tool_calls=None))]
            self.model = "gpt-4"
            self._hidden_params = {}

    headers = await callback.async_post_call_response_headers_hook(
        data, None, MockResponse(), request_headers=req_headers
    )
    
    assert headers == {"x-open-llm-proxy-served-by": "openai/gpt-4"}
    assert global_attribution_store.get(uid) == "openai/gpt-4"

def test_internal_attribution_endpoint(monkeypatch, tmp_path):
    app = FastAPI()
    register_attribution_endpoint(app)
    client = TestClient(app, client=("127.0.0.1", 50000))
    uid = str(uuid.uuid4())
    
    # 1. No token env set
    monkeypatch.delenv("OPEN_LLM_PROXY_ATTRIBUTION_TOKEN", raising=False)
    response = client.get(f"/internal/attribution/v1/{uid}", headers={"Authorization": f"Bearer secret"})
    assert response.status_code == 401

    # 2. Configured but unauthorized due to missing header
    monkeypatch.setenv("OPEN_LLM_PROXY_ATTRIBUTION_TOKEN", "super-secret-token")
    response = client.get(f"/internal/attribution/v1/{uid}")
    assert response.status_code == 401
    
    # 3. Unauthorized due to incorrect token
    response = client.get(f"/internal/attribution/v1/{uid}", headers={"Authorization": f"Bearer bad-token"})
    assert response.status_code == 401

    # 4. Authorized but ID doesn't exist
    response = client.get(f"/internal/attribution/v1/{uid}", headers={"Authorization": f"Bearer super-secret-token"})
    assert response.status_code == 404

    # 5. Invalid UUID format
    response = client.get(f"/internal/attribution/v1/not-a-uuid", headers={"Authorization": f"Bearer super-secret-token"})
    assert response.status_code == 404

    # 6. Active record retrieval
    global_attribution_store.clear()
    global_attribution_store.set(uid, "test-model-served")
    
    response = client.get(f"/internal/attribution/v1/{uid}", headers={"Authorization": f"Bearer super-secret-token"})
    assert response.status_code == 200
    assert response.json() == {"servedBy": "test-model-served"}
    assert response.headers.get("Cache-Control") == "no-store"
