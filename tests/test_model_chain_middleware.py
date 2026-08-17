import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from open_llm_proxy.model_chain_middleware import (
    install_model_chain_middleware,
    rewrite_model_chain_body,
)


@pytest.mark.anyio
async def test_rewrite_model_chain_middleware():
    app = FastAPI()

    # Invoke the exported installer
    install_model_chain_middleware(app)

    @app.post("/v1/chat/completions")
    async def dummy_endpoint(request: Request):
        body = await request.json()
        return body

    @app.post("/other/endpoint")
    async def other_dummy_endpoint(request: Request):
        body = await request.json()
        return body

    client = TestClient(app)

    # Case 1: Bracketed with open-llm-proxy/ prefix and commas -> should rewrite to semicolons
    payload = {
        "model": "open-llm-proxy/[claude-cli/claude-sonnet-5,github-copilot/claude-sonnet-5]",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert response.json()["model"] == "[claude-cli/claude-sonnet-5;github-copilot/claude-sonnet-5]"

    # Case 2: Bracketed without prefix -> should rewrite
    payload = {
        "model": "[claude-cli/claude-sonnet-5,github-copilot/claude-sonnet-5]",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert response.json()["model"] == "[claude-cli/claude-sonnet-5;github-copilot/claude-sonnet-5]"

    # Case 3: No bracket (not a chain) -> unchanged
    payload = {
        "model": "open-llm-proxy/claude-cli/claude-sonnet-5",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert response.json()["model"] == "open-llm-proxy/claude-cli/claude-sonnet-5"

    # Case 4: Non-target path or non-POST -> unchanged
    response = client.post(
        "/other/endpoint",
        json={
            "model": "open-llm-proxy/[claude-cli/claude-sonnet-5,github-copilot/claude-sonnet-5]"
        },
    )
    assert response.status_code == 200
    assert (
        response.json()["model"]
        == "open-llm-proxy/[claude-cli/claude-sonnet-5,github-copilot/claude-sonnet-5]"
    )

    # Case 5: malformed target body is replayed unchanged.
    assert rewrite_model_chain_body(b"not-json") == b"not-json"

    # Case 6: existing internal alias is unchanged.
    internal = b'{"model":"[a/b;c/d]"}'
    assert rewrite_model_chain_body(internal) == internal

    # Case 7: @account tokens in a chain round-trip with @ preserved and commas replaced by semicolons
    payload = {
        "model": "open-llm-proxy/[claude-cli@work/claude-opus-4-8,claude-cli@home/claude-opus-4-8,github-copilot/claude-opus-4.8]",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert response.json()["model"] == (
        "[claude-cli@work/claude-opus-4-8;claude-cli@home/claude-opus-4-8;github-copilot/claude-opus-4.8]"
    )
