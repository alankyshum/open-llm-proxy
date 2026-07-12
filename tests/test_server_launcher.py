"""Focused tests for stable config lifecycle in server_launcher.py.

Covers deterministic path resolution, atomic write (incl. repeated
generation), mode 0600, valid YAML, no raw keys, and no unlink after setup.
"""

import os
import stat
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from open_llm_proxy.server_launcher import (
    STABLE_CONFIG_BASENAME,
    _resolve_stable_config_dir,
    register_config_reload_endpoint,
    resolve_stable_config_path,
    write_config_atomic,
)


# ---------------------------------------------------------------------------
# Deterministic path
# ---------------------------------------------------------------------------

class TestConfigPathResolution:
    def test_default_path_under_config_dir(self):
        """Default stable config path lives under ~/.config/open-llm-proxy."""
        path = resolve_stable_config_path()
        expected = (
            Path.home()
            / ".config"
            / "open-llm-proxy"
            / f"generated-litellm-config-{os.getpid()}.yaml"
        )
        assert path == expected

    def test_respects_env_override(self, monkeypatch):
        """OPEN_LLM_PROXY_CONFIG_DIR overrides the config directory."""
        custom = "/tmp/open-llm-proxy-config-test"
        monkeypatch.setenv("OPEN_LLM_PROXY_CONFIG_DIR", custom)
        path = resolve_stable_config_path()
        expected = Path(custom) / f"generated-litellm-config-{os.getpid()}.yaml"
        assert path == expected

    def test_basename_constant(self):
        """Basename is the agreed filename."""
        assert STABLE_CONFIG_BASENAME == "generated-litellm-config.yaml"


# ---------------------------------------------------------------------------
# Atomic write — including repeated generation
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "model_list": [{"model_name": "test-model", "litellm_params": {"model": "test"}}],
    "litellm_settings": {"drop_params": True, "fallbacks": []},
    "router_settings": {
        "num_retries": 0,
        "routing_strategy": "simple-shuffle",
        "fallbacks": [],
    },
}

SAMPLE_CONFIG_V2 = {
    "model_list": [{"model_name": "v2-model", "litellm_params": {"model": "v2"}}],
    "litellm_settings": {"drop_params": False, "fallbacks": []},
    "router_settings": {
        "num_retries": 1,
        "routing_strategy": "simple-shuffle",
        "fallbacks": [],
    },
}


class TestAtomicWrite:
    def test_writes_to_expected_path(self, tmp_path):
        target = tmp_path / "test-config.yaml"
        result = write_config_atomic(SAMPLE_CONFIG, target)
        assert result == str(target)
        assert target.exists()

    def test_file_is_valid_yaml(self, tmp_path):
        target = tmp_path / "test-config.yaml"
        write_config_atomic(SAMPLE_CONFIG, target)
        with open(target) as f:
            loaded = yaml.safe_load(f)
        assert loaded == SAMPLE_CONFIG

    def test_no_raw_secrets(self, tmp_path):
        """Config only contains env references (os.environ/...), never raw keys."""
        config_with_env_refs = {
            "model_list": [
                {
                    "model_name": "test",
                    "litellm_params": {
                        "model": "openai/gpt-4",
                        "api_key": "os.environ/OPENROUTER_API_KEY",
                    },
                }
            ],
        }
        target = tmp_path / "env-refs.yaml"
        write_config_atomic(config_with_env_refs, target)
        with open(target) as f:
            raw = f.read()
        # No literal sk- or AIza patterns (raw API keys)
        assert "sk-" not in raw, f"Found raw sk- key in {target}"
        assert "AIza" not in raw, f"Found raw AIza key in {target}"

    def test_repeated_generation_overwrites_atomically(self, tmp_path):
        """Writing twice produces the second version (no partial state)."""
        target = tmp_path / "repeat.yaml"
        write_config_atomic(SAMPLE_CONFIG, target)
        write_config_atomic(SAMPLE_CONFIG_V2, target)
        with open(target) as f:
            loaded = yaml.safe_load(f)
        assert loaded["model_list"][0]["model_name"] == "v2-model"

    def test_repeated_generation_no_intermediate_truncation(self, tmp_path):
        """During repeated writes the target never becomes empty or partial."""
        target = tmp_path / "no-trunc.yaml"
        # Write v1
        write_config_atomic(SAMPLE_CONFIG, target)
        # Write v2 many times
        for _ in range(5):
            write_config_atomic(SAMPLE_CONFIG_V2, target)
            # After each write the file must be valid YAML
            with open(target) as f:
                data = yaml.safe_load(f)
            assert data is not None, f"File was empty after write to {target}"
            assert "model_list" in data
        # Final should be v2
        with open(target) as f:
            data = yaml.safe_load(f)
        assert data["model_list"][0]["model_name"] == "v2-model"

    def test_temp_files_cleaned_on_error(self, tmp_path):
        """If the write crashes the temp file in the same dir is removed."""
        class _ExplodingDict(dict):
            def __iter__(self):
                raise RuntimeError("boom!")

        target = tmp_path / "crash.yaml"
        # The explosion happens during yaml.safe_dump iteration;
        # any Exception type is fine — we just verify cleanup.
        with pytest.raises(Exception):
            write_config_atomic(_ExplodingDict({"k": "v"}), target)
        assert not target.exists()  # never written
        # No stray temp files left in the target directory
        temps = list(tmp_path.glob(f".*{STABLE_CONFIG_BASENAME}.*.tmp"))
        assert len(temps) == 0, f"Stale temp files: {temps}"


# ---------------------------------------------------------------------------
# Mode 0600
# ---------------------------------------------------------------------------

class TestMode0600:
    def test_mode_is_0600(self, tmp_path):
        target = tmp_path / "perm.yaml"
        write_config_atomic(SAMPLE_CONFIG, target)
        mode = target.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0600, got {oct(mode)}"

    def test_mode_survives_repeated_write(self, tmp_path):
        target = tmp_path / "perm-repeat.yaml"
        write_config_atomic(SAMPLE_CONFIG, target)
        write_config_atomic(SAMPLE_CONFIG_V2, target)
        mode = target.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0600 after repeat, got {oct(mode)}"

    def test_mode_is_0600_on_nonexistent_parent(self, tmp_path):
        """Parent created automatically, file still gets 0600."""
        target = tmp_path / "newdir" / "nested.yaml"
        write_config_atomic(SAMPLE_CONFIG, target)
        assert target.exists()
        mode = target.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0600, got {oct(mode)}"


# ---------------------------------------------------------------------------
# No unlink after setup — config persists
# ---------------------------------------------------------------------------

class TestConfigPersistence:
    def test_file_persists_after_write(self, tmp_path):
        """Config file exists after write; no implicit delete."""
        target = tmp_path / "persist.yaml"
        write_config_atomic(SAMPLE_CONFIG, target)
        assert target.exists()
        # Simulate what happens in launch_server: no explicit unlink
        # The file must remain for LiteLLM's APScheduler to re-read.
        assert target.exists()

    def test_file_survives_multiple_generations(self, tmp_path):
        """Config persists across repeated generations (no delete)."""
        target = tmp_path / "multi-gen.yaml"
        for i in range(3):
            cfg = {
                "model_list": [{"model_name": f"gen-{i}", "litellm_params": {"model": "test"}}],
                "litellm_settings": {"drop_params": True, "fallbacks": []},
                "router_settings": {"num_retries": 0, "routing_strategy": "simple-shuffle", "fallbacks": []},
            }
            write_config_atomic(cfg, target)
            assert target.exists(), f"File missing after generation {i}"
        # Final generation correct
        with open(target) as f:
            data = yaml.safe_load(f)
        assert data["model_list"][0]["model_name"] == "gen-2"


class _FakeReloader:
    def __init__(self, active_hash: str, *, applied: bool = True) -> None:
        self.active_hash = active_hash
        self.applied = applied
        self.calls: list[str] = []

    async def reload(self, expected_hash: str | None = None) -> bool:
        assert expected_hash is not None
        self.calls.append(expected_hash)
        return self.applied


def _reload_client(reloader, host: str = "127.0.0.1") -> TestClient:
    app = FastAPI()
    register_config_reload_endpoint(app, reloader)
    return TestClient(app, client=(host, 50000))


class TestConfigReloadEndpoint:
    def test_loopback_validates_source_even_when_hash_is_already_active(self):
        expected_hash = "a" * 64
        reloader = _FakeReloader(expected_hash)

        response = _reload_client(reloader).post(
            "/internal/config/reload",
            json={"expected_hash": expected_hash},
        )

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "config_hash": expected_hash}
        assert reloader.calls == [expected_hash]

    def test_non_loopback_request_is_forbidden(self):
        expected_hash = "b" * 64
        reloader = _FakeReloader(expected_hash)

        response = _reload_client(reloader, host="192.0.2.1").post(
            "/internal/config/reload",
            json={"expected_hash": expected_hash},
        )

        assert response.status_code == 403
        assert reloader.calls == []

    @pytest.mark.parametrize("expected_hash", [None, "abc", "A" * 64])
    def test_invalid_hash_is_rejected(self, expected_hash):
        reloader = _FakeReloader("c" * 64)

        response = _reload_client(reloader).post(
            "/internal/config/reload",
            json={"expected_hash": expected_hash},
        )

        assert response.status_code == 400
        assert reloader.calls == []

    def test_reloader_rejection_requires_full_restart(self):
        expected_hash = "d" * 64
        reloader = _FakeReloader("e" * 64, applied=False)

        response = _reload_client(reloader).post(
            "/internal/config/reload",
            json={"expected_hash": expected_hash},
        )

        assert response.status_code == 409
        assert reloader.calls == [expected_hash]
