from __future__ import annotations

import asyncio
import httpx
from typing import Tuple

from open_llm_proxy import openrouter_creds, opencode_creds, creds, anthropic_client, copilot_creds

def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)

def check_provider(provider: str) -> Tuple[bool, str]:
    """
    Perform a synchronous, read-only connectivity probe on a given LLM provider.
    5s timeout, no retries, returns a static status string.
    """
    prov = provider.lower().replace("-", "_").strip()
    
    url = ""
    headers = {}

    try:
        if prov == "openrouter":
            key = openrouter_creds.get_persisted_api_key()
            if not key:
                return False, "Missing Credentials"
            url = "https://openrouter.ai/api/v1/auth/key"
            headers = {
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
            }
        elif prov == "opencode":
            key = opencode_creds.get_opencode_api_key()
            if not key:
                return False, "Missing Credentials"
            url = "https://opencode.ai/zen/v1/models"
            headers = {
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
            }
        elif prov in ("claude", "anthropic", "claude_cli"):
            key = creds.get_api_key()
            if not key:
                return False, "Missing Credentials"
            url = "https://api.anthropic.com/v1/models"
            headers = dict(anthropic_client._headers(key))
            headers["accept"] = "application/json"
        elif prov in ("copilot", "github_copilot"):
            token, base_url = _run_async(copilot_creds.get_copilot_token())
            if not token or not base_url:
                return False, "Missing Credentials"
            url = f"{base_url}/models"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": copilot_creds.USER_AGENT,
                "Editor-Version": copilot_creds.EDITOR_VERSION,
                "Editor-Plugin-Version": copilot_creds.EDITOR_PLUGIN_VERSION,
                "Copilot-Integration-Id": copilot_creds.COPILOT_INTEGRATION_ID,
                "X-GitHub-Api-Version": copilot_creds.X_GITHUB_API_VERSION,
            }
        else:
            raise ValueError(f"Unknown provider: {provider}")
    except Exception:
        return False, "Missing Credentials"

    transport = httpx.HTTPTransport(retries=0)
    try:
        with httpx.Client(transport=transport, timeout=5.0) as client:
            resp = client.get(url, headers=headers)
            
            if 200 <= resp.status_code < 300:
                return True, "Ready"
            elif resp.status_code in (401, 403):
                return False, "Authentication Failed"
            elif resp.status_code == 429:
                return False, "Rate Limited"
            elif 400 <= resp.status_code < 500:
                return False, "Client Error"
            elif 500 <= resp.status_code < 600:
                return False, "Server Error"
            else:
                return False, "Unexpected Status"
    except httpx.TimeoutException:
        return False, "Timeout"
    except httpx.RequestError:
        return False, "Connection Failed"
    except Exception:
        return False, "Connection Failed"
