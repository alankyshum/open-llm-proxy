"""In-process routing config reload for the LiteLLM proxy."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litellm

from open_llm_proxy.config_gen import (
    configured_model_tokens_from_data,
    generate_config_from_data,
    parse_agent_config,
)

log = logging.getLogger("open_llm_proxy.reloader")

_HOT_ROUTER_SETTINGS = ("num_retries", "routing_strategy", "fallbacks")


def config_sha256(path: str | Path) -> str:
    """Return SHA-256 of exact source config bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rate_limit_policy(data: dict) -> Any:
    return copy.deepcopy(data.get("rate_limit_policy"))


@dataclass(frozen=True)
class _PreparedReload:
    source_hash: str
    policy: Any
    config: dict[str, Any]
    model_tokens: set[str]


class ConfigReloader:
    """Atomically replace live Router routes while preserving in-flight calls."""

    def __init__(
        self,
        *,
        source_path: str | Path,
        generated_path: str | Path,
        initial_source: bytes,
        initial_config: dict[str, Any],
        write_config: Callable[[dict[str, Any], str | Path], str],
        rate_limit_callback: Any = None,
    ) -> None:
        self.source_path = Path(source_path)
        self.generated_path = Path(generated_path)
        self._write_config = write_config
        self._rate_limit_callback = rate_limit_callback
        initial_data = parse_agent_config(initial_source)
        self._active_hash = hashlib.sha256(initial_source).hexdigest()
        self._active_policy = _rate_limit_policy(initial_data)
        self._active_config = copy.deepcopy(initial_config)
        self._reload_lock = asyncio.Lock()

    @property
    def active_hash(self) -> str:
        return self._active_hash

    def _runtime(self) -> tuple[Any, Any]:
        from litellm.proxy import proxy_server

        router = proxy_server.llm_router
        if router is None:
            raise RuntimeError("LiteLLM Router is not initialized")
        return proxy_server, router

    def _prepare_reload(self) -> _PreparedReload:
        """Read and generate from exactly one source snapshot."""
        source = self.source_path.read_bytes()
        data = parse_agent_config(source)
        return _PreparedReload(
            source_hash=hashlib.sha256(source).hexdigest(),
            policy=_rate_limit_policy(data),
            config=generate_config_from_data(data),
            model_tokens=configured_model_tokens_from_data(data),
        )

    async def reload(self, expected_hash: str | None = None) -> bool:
        """Apply current source config. Retain old routes on any error."""
        async with self._reload_lock:
            staged_path: Path | None = None
            try:
                prepared = await asyncio.to_thread(self._prepare_reload)
                if expected_hash is not None and prepared.source_hash != expected_hash:
                    log.warning(
                        "Config hot reload rejected; expected hash %s, found %s",
                        expected_hash,
                        prepared.source_hash,
                    )
                    return False
                if prepared.source_hash == self._active_hash:
                    log.info("Config hot reload skipped; source hash unchanged")
                    return True
                if prepared.policy != self._active_policy:
                    raise RuntimeError("rate_limit_policy changes require a full restart")

                callback_store = getattr(self._rate_limit_callback, "store", None)
                if callback_store is not None:
                    await asyncio.to_thread(callback_store.register_models, prepared.model_tokens)
                staged_path = await asyncio.to_thread(self._stage_config, prepared.config)
                return self._apply(prepared, staged_path)
            except Exception:
                log.exception("Config hot reload failed; existing routes remain active")
                return False
            finally:
                if staged_path is not None:
                    staged_path.unlink(missing_ok=True)

    def _stage_config(self, config: dict[str, Any]) -> Path:
        """Write generated config beside its target before touching live routes."""
        self.generated_path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            prefix=f".{self.generated_path.name}.reload-",
            suffix=".tmp",
            dir=self.generated_path.parent,
        )
        os.close(fd)
        staged_path = Path(name)
        try:
            self._write_config(config, staged_path)
            return staged_path
        except BaseException:
            staged_path.unlink(missing_ok=True)
            raise

    def _apply(self, prepared: _PreparedReload, staged_path: Path) -> bool:
        """Swap live Router state and publish the pre-staged config."""
        new_config = prepared.config
        proxy_server, router = self._runtime()
        old_models = copy.deepcopy(router.model_list)
        old_settings = copy.deepcopy(router.get_settings())
        old_proxy_models = copy.deepcopy(getattr(proxy_server, "llm_model_list", None))
        old_fallbacks = copy.deepcopy(getattr(litellm, "fallbacks", None))
        old_drop_params = getattr(litellm, "drop_params", None)
        try:
            new_models = copy.deepcopy(new_config.get("model_list") or [])
            if not new_models:
                raise ValueError("generated config contains no models")
            router.set_model_list(new_models)

            generated_router_settings = new_config.get("router_settings") or {}
            router.update_settings(
                **{
                    key: copy.deepcopy(generated_router_settings[key])
                    for key in _HOT_ROUTER_SETTINGS
                    if key in generated_router_settings
                }
            )

            expected_names = {model["model_name"] for model in new_models}
            actual_names = set(router.get_model_names())
            if actual_names != expected_names:
                raise RuntimeError(
                    f"Router model verification failed: expected {expected_names}, got {actual_names}"  # intentional long protocol text or compatibility message  # noqa: E501
                )

            proxy_server.llm_model_list = copy.deepcopy(new_models)
            generated_litellm_settings = new_config.get("litellm_settings") or {}
            if "fallbacks" in generated_litellm_settings:
                litellm.fallbacks = copy.deepcopy(generated_litellm_settings["fallbacks"])
            if "drop_params" in generated_litellm_settings:
                litellm.drop_params = generated_litellm_settings["drop_params"]

            os.replace(staged_path, self.generated_path)
        except Exception:
            self._rollback(
                proxy_server=proxy_server,
                router=router,
                models=old_models,
                settings=old_settings,
                proxy_models=old_proxy_models,
                fallbacks=old_fallbacks,
                drop_params=old_drop_params,
            )
            raise

        self._active_config = copy.deepcopy(new_config)
        self._active_policy = prepared.policy
        self._active_hash = prepared.source_hash
        log.info("Config hot reload applied: %s", prepared.source_hash)
        return True

    def _rollback(
        self,
        *,
        proxy_server: Any,
        router: Any,
        models: list[dict[str, Any]],
        settings: dict[str, Any],
        proxy_models: Any,
        fallbacks: Any,
        drop_params: Any,
    ) -> None:
        try:
            router.set_model_list(models)
            router.update_settings(**settings)
            proxy_server.llm_model_list = proxy_models
            litellm.fallbacks = fallbacks
            litellm.drop_params = drop_params
        except Exception:
            log.exception("Config hot reload rollback failed")
