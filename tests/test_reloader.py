from __future__ import annotations

import copy
import asyncio
import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace

import litellm
import yaml

from open_llm_proxy.config_gen import generate_config
from open_llm_proxy.reloader import ConfigReloader, config_sha256


def _source(model: str = "vendor/model-a", *, policy: dict | None = None) -> dict:
    return {
        "rate_limit_policy": policy or {},
        "file_settings": {
            "opencode": {
                "model": f"open-llm-proxy/{model}",
                "supported_models": [model],
            }
        },
        "agents": {},
    }


def _write_source(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


class _FakeRouter:
    def __init__(self, config: dict) -> None:
        self.model_list = copy.deepcopy(config["model_list"])
        self.settings = {
            "num_retries": config["router_settings"]["num_retries"],
            "routing_strategy": config["router_settings"]["routing_strategy"],
            "fallbacks": copy.deepcopy(config["router_settings"]["fallbacks"]),
            "routing_groups": [],
        }
        self.set_calls = 0

    def set_model_list(self, models: list[dict]) -> None:
        self.set_calls += 1
        self.model_list = copy.deepcopy(models)

    def update_settings(self, **settings) -> None:
        self.settings.update(copy.deepcopy(settings))

    def get_settings(self) -> dict:
        return copy.deepcopy(self.settings)

    def get_model_names(self) -> list[str]:
        return sorted({model["model_name"] for model in self.model_list})


class _FakeStore:
    def __init__(self) -> None:
        self.registered: set[str] | None = None

    def register_models(self, models: set[str]) -> None:
        self.registered = set(models)


def _controller(tmp_path: Path):
    source_path = tmp_path / "agent-config.yml"
    generated_path = tmp_path / "generated.yml"
    _write_source(source_path, _source())
    initial = generate_config(str(source_path))
    initial_source = source_path.read_bytes()
    generated_path.write_text(yaml.safe_dump(initial, sort_keys=False))
    router = _FakeRouter(initial)
    proxy_server = SimpleNamespace(llm_model_list=copy.deepcopy(initial["model_list"]))
    store = _FakeStore()
    callback = SimpleNamespace(store=store)

    def writer(config: dict, path: str | Path) -> str:
        Path(path).write_text(yaml.safe_dump(config, sort_keys=False))
        return str(path)

    controller = ConfigReloader(
        source_path=source_path,
        generated_path=generated_path,
        initial_source=initial_source,
        initial_config=initial,
        write_config=writer,
        rate_limit_callback=callback,
    )
    controller._runtime = lambda: (proxy_server, router)
    return controller, source_path, generated_path, proxy_server, router, store


def test_config_sha256_hashes_exact_bytes(tmp_path):
    path = tmp_path / "config.yml"
    path.write_bytes(b"exact bytes\n")
    assert config_sha256(path) == hashlib.sha256(b"exact bytes\n").hexdigest()


def test_reload_updates_routes_globals_file_and_hash(tmp_path, monkeypatch):
    controller, source, generated, proxy_server, router, store = _controller(tmp_path)
    old_hash = controller.active_hash
    monkeypatch.setattr(litellm, "fallbacks", ["old"], raising=False)
    monkeypatch.setattr(litellm, "drop_params", False, raising=False)

    _write_source(source, _source("vendor/model-b"))

    assert asyncio.run(controller.reload()) is True
    assert controller.active_hash != old_hash
    assert controller.active_hash == config_sha256(source)
    assert "vendor/model-b" in router.get_model_names()
    assert proxy_server.llm_model_list == router.model_list
    assert store.registered == {"vendor/model-b"}
    assert yaml.safe_load(generated.read_text())["model_list"] == router.model_list
    assert litellm.drop_params is True


def test_unchanged_reload_is_noop(tmp_path):
    controller, _source_path, _generated, _proxy, router, _store = _controller(tmp_path)

    assert asyncio.run(controller.reload()) is True
    assert router.set_calls == 0


def test_expected_hash_mismatch_rejects_reload(tmp_path):
    controller, source, _generated, _proxy, router, _store = _controller(tmp_path)
    old_hash = controller.active_hash
    _write_source(source, _source("vendor/model-b"))

    assert asyncio.run(controller.reload(expected_hash="0" * 64)) is False
    assert controller.active_hash == old_hash
    assert router.set_calls == 0


def test_rate_limit_policy_change_requires_restart(tmp_path):
    controller, source, _generated, proxy_server, router, _store = _controller(tmp_path)
    old_hash = controller.active_hash
    old_models = copy.deepcopy(router.model_list)
    old_proxy_models = copy.deepcopy(proxy_server.llm_model_list)
    _write_source(source, _source(policy={"database": "/tmp/new.sqlite"}))

    assert asyncio.run(controller.reload()) is False
    assert controller.active_hash == old_hash
    assert router.model_list == old_models
    assert proxy_server.llm_model_list == old_proxy_models


def test_failed_write_rolls_back_router_and_globals(tmp_path, monkeypatch):
    controller, source, generated, proxy_server, router, _store = _controller(tmp_path)
    old_hash = controller.active_hash
    old_models = copy.deepcopy(router.model_list)
    old_proxy_models = copy.deepcopy(proxy_server.llm_model_list)
    old_generated = generated.read_text()
    monkeypatch.setattr(litellm, "fallbacks", ["old"], raising=False)
    monkeypatch.setattr(litellm, "drop_params", False, raising=False)
    writes = 0

    def failing_writer(config: dict, path: str | Path) -> str:
        nonlocal writes
        writes += 1
        if writes == 1:
            raise OSError("disk full")
        Path(path).write_text(yaml.safe_dump(config, sort_keys=False))
        return str(path)

    controller._write_config = failing_writer
    _write_source(source, _source("vendor/model-b"))

    assert asyncio.run(controller.reload()) is False
    assert controller.active_hash == old_hash
    assert router.model_list == old_models
    assert proxy_server.llm_model_list == old_proxy_models
    assert litellm.fallbacks == ["old"]
    assert litellm.drop_params is False
    assert generated.read_text() == old_generated
    assert router.set_calls == 0


def test_concurrent_reload_waits_and_revalidates(tmp_path):
    controller, source, _generated, _proxy, _router, _store = _controller(tmp_path)
    _write_source(source, _source("vendor/model-b"))
    expected_hash = config_sha256(source)
    original_prepare = controller._prepare_reload
    first_started = threading.Event()
    release_first = threading.Event()
    prepare_calls = 0

    def slow_first_prepare():
        nonlocal prepare_calls
        prepare_calls += 1
        if prepare_calls == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        return original_prepare()

    controller._prepare_reload = slow_first_prepare

    async def scenario():
        first = asyncio.create_task(controller.reload(expected_hash=expected_hash))
        assert await asyncio.to_thread(first_started.wait, 2)
        second = asyncio.create_task(controller.reload(expected_hash=expected_hash))
        await asyncio.sleep(0)
        assert not second.done()
        release_first.set()
        assert await first is True
        assert await second is True

    asyncio.run(scenario())
    assert prepare_calls == 2


def test_prepare_reload_uses_one_immutable_snapshot(tmp_path):
    controller, source, _generated, _proxy, _router, _store = _controller(tmp_path)
    _write_source(source, _source("vendor/model-b"))

    prepared = controller._prepare_reload()
    _write_source(source, _source("vendor/model-c"))

    names = {model["model_name"] for model in prepared.config["model_list"]}
    assert "vendor/model-b" in names
    assert "vendor/model-c" not in names
    assert prepared.source_hash != config_sha256(source)
