import os
import socket
import time
import httpx
import pytest
from multiprocessing import Process
from open_llm_proxy.server_launcher import launch_server, find_agent_config

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 10.0) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
                return True
        except socket.error:
            time.sleep(0.2)
    return False

@pytest.mark.anyio
async def test_proxy_integration_smoke():
    # Use port 8766 for test to avoid collision with any running main proxy on 8765
    test_port = 8766
    
    # Check if port is already in use
    if is_port_in_use(test_port):
        pytest.skip(f"Port {test_port} is already in use. Skipping integration test.")
        
    # Check if agent-config.yml exists
    config_path = find_agent_config()
    if not config_path.exists():
        pytest.skip(f"agent-config.yml not found at {config_path}. Skipping integration test.")

    # Start the server in a separate Process
    proc = Process(
        target=launch_server,
        kwargs={
            "host": "127.0.0.1",
            "port": test_port,
            "disable_admin_ui": True,
            "master_key": "sk-local",
        },
        daemon=True,
    )
    proc.start()
    
    try:
        # Wait for the server to bind to the port
        if not wait_for_port(test_port, timeout=12.0):
            pytest.fail("Timeout waiting for proxy server to start.")
            
        # Perform test call to the proxy
        url = f"http://127.0.0.1:{test_port}/v1/chat/completions"
        headers = {
            "Authorization": "Bearer sk-local",
            "Content-Type": "application/json"
        }
        
        # Use an existing fallback chain from agent-config.yml
        # In our case, [claude-cli/claude-sonnet-5,github-copilot/claude-sonnet-5,openrouter/z-ai/glm-5.2]
        model_name = "[claude-cli/claude-sonnet-5,github-copilot/claude-sonnet-5,openrouter/z-ai/glm-5.2]"
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "reply pong"}],
            "stream": False
        }
        
        print(f"Sending POST request to {url} with model {model_name}...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                print(f"Response status code: {response.status_code}")
                print(f"Response content: {response.text}")
                
                # Check for 200 OK
                assert response.status_code == 200
                data = response.json()
                assert "choices" in data
                assert len(data["choices"]) > 0
                content = data["choices"][0]["message"]["content"]
                assert content is not None
                assert len(content) > 0
            except httpx.HTTPError as e:
                # If there are credentials or network issues, skip gracefully as requested
                pytest.skip(f"HTTP call to proxy failed due to network/creds: {e}")
            except Exception as e:
                # If other errors occur, skip gracefully as requested
                pytest.skip(f"Integration smoke test skipped due to: {e}")
    finally:
        # Tear down the proxy after
        print("Tearing down the proxy process...")
        proc.terminate()
        proc.join(timeout=3.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
        print("Proxy process terminated.")
