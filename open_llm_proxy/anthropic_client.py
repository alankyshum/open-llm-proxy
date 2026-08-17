from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from open_llm_proxy import creds
from open_llm_proxy.errors import RateLimitError

log = logging.getLogger("open_llm_proxy.anthropic_client")

API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0),
            limits=httpx.Limits(max_keepalive_connections=32, max_connections=64),
            http2=True,
        )
    return _client


async def startup() -> None:
    _get_client()


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _headers(key: str) -> dict[str, str]:
    headers = {
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
        "accept": "text/event-stream",
    }
    if key.startswith("sk-ant-oat01-"):
        headers["Authorization"] = f"Bearer {key}"
        headers["anthropic-beta"] = "oauth-2025-04-20"
    else:
        headers["x-api-key"] = key
    return headers


async def stream_messages(
    payload: dict[str, Any], *, account: str | None = None
) -> AsyncIterator[
    tuple[str, dict[str, Any]]
]:  # intentional long protocol text or compatibility message
    client = _get_client()
    attempts = 2
    for attempt in range(attempts):
        retry_needed = False
        key = creds.get_api_key(account=account)
        headers = _headers(key)

        async with client.stream("POST", API_URL, headers=headers, json=payload) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", errors="replace")[:2000]
                if resp.status_code == 401 and attempt < attempts - 1:
                    creds.clear_cache(account=account)
                    creds.refresh_anthropic_oauth(stale_token=key, account=account)
                    retry_needed = True
                else:
                    if resp.status_code == 429:
                        retry_after_str = resp.headers.get("retry-after")
                        retry_after: float | None = None
                        if retry_after_str is not None:
                            try:
                                retry_after = float(retry_after_str)
                            except ValueError:
                                pass
                        raise RateLimitError(
                            f"Anthropic API error 429: {body}",
                            retry_after=retry_after,
                            headers=dict(resp.headers),
                        )
                    raise RuntimeError(f"Anthropic API error {resp.status_code}: {body}")

            if retry_needed:
                continue

            current_event: str | None = None
            async for line in resp.aiter_lines():
                if not line:
                    current_event = None
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    current_event = line[len("event:") :].strip()
                    continue
                if line.startswith("data:"):
                    if current_event in ("ping", None):
                        continue
                    payload_str = line[len("data:") :].strip()
                    if not payload_str:
                        continue
                    try:
                        data = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    yield current_event, data
            break


async def send_messages(payload: dict[str, Any], *, account: str | None = None) -> dict[str, Any]:
    client = _get_client()
    body = dict(payload)
    body["stream"] = False
    attempts = 2
    for attempt in range(attempts):
        key = creds.get_api_key(account=account)
        headers = _headers(key)
        headers["accept"] = "application/json"

        resp = await client.post(API_URL, headers=headers, json=body)
        if resp.status_code >= 400:
            if resp.status_code == 401 and attempt < attempts - 1:
                creds.clear_cache(account=account)
                creds.refresh_anthropic_oauth(stale_token=key, account=account)
                continue
            if resp.status_code == 429:
                retry_after_str = resp.headers.get("retry-after")
                retry_after: float | None = None
                if retry_after_str is not None:
                    try:
                        retry_after = float(retry_after_str)
                    except ValueError:
                        pass
                raise RateLimitError(
                    f"Anthropic API error 429: {resp.text}",
                    retry_after=retry_after,
                    headers=dict(resp.headers),
                )
            raise RuntimeError(f"Anthropic API error {resp.status_code}: {resp.text}")
        return resp.json()


async def fetch_models(*, account: str | None = None) -> list[dict[str, Any]]:
    client = _get_client()
    url = "https://api.anthropic.com/v1/models"

    models: list[dict[str, Any]] = []
    has_more = True
    after_id: str | None = None
    refreshed_at_least_once = False

    while has_more:
        params = {}
        if after_id:
            params["after_id"] = after_id

        attempts = 1 if refreshed_at_least_once else 2
        for attempt in range(attempts):
            key = creds.get_api_key(account=account)
            headers = _headers(key)
            headers["accept"] = "application/json"

            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code >= 400:
                if resp.status_code == 401 and attempt == 0 and not refreshed_at_least_once:
                    creds.clear_cache(account=account)
                    creds.refresh_anthropic_oauth(stale_token=key, account=account)
                    refreshed_at_least_once = True
                    continue
                if resp.status_code == 429:
                    body = resp.text[:2000]
                    retry_after_str = resp.headers.get("retry-after")
                    retry_after = None
                    if retry_after_str is not None:
                        try:
                            retry_after = float(retry_after_str)
                        except ValueError:
                            pass
                    raise RateLimitError(
                        f"Anthropic API error 429: {body}",
                        retry_after=retry_after,
                        headers=dict(resp.headers),
                    )
                raise RuntimeError(f"Anthropic API error {resp.status_code}: {resp.text}")
            break

        data = resp.json()
        models.extend(data.get("data") or [])
        has_more = data.get("has_more", False)
        if has_more:
            after_id = data.get("last_id")
            if not after_id:
                break
        else:
            break

    return models
