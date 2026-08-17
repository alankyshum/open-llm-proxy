from __future__ import annotations

import os
from pathlib import Path

import pytest

from open_llm_proxy import env_creds


@pytest.fixture
def cfg_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set OLP_CONFIG_DIR to a temp directory."""
    d = tmp_path / "olp_config"
    d.mkdir()
    monkeypatch.setenv("OLP_CONFIG_DIR", str(d))
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    return d


class TestSetEnvKey:
    def test_round_trip(self, cfg_dir):
        env_creds.set_env_key("TEST_KEY", "test-value")
        assert env_creds.get_env_key("TEST_KEY") == "test-value"

    def test_file_permissions_0600(self, cfg_dir):
        env_creds.set_env_key("TEST_KEY", "secret")
        env_file = cfg_dir / "env"
        assert env_file.is_file()
        perms = os.stat(str(env_file)).st_mode & 0o777
        assert perms <= 0o600, f"Expected 0600 or tighter, got {oct(perms)}"
        assert perms >= 0o600, f"Expected at least 0600, got {oct(perms)}"

    def test_preserves_unrelated_lines(self, cfg_dir):
        env_file = cfg_dir / "env"
        env_file.write_text("UNRELATED=keep\n", encoding="utf-8")
        env_creds.set_env_key("TEST_KEY", "new-val")
        content = env_file.read_text(encoding="utf-8")
        assert "UNRELATED=keep" in content
        assert "TEST_KEY=" in content

    def test_updates_existing_key(self, cfg_dir):
        env_creds.set_env_key("TEST_KEY", "first")
        env_creds.set_env_key("TEST_KEY", "second")
        assert env_creds.get_env_key("TEST_KEY") == "second"

    def test_empty_value_raises(self, cfg_dir):
        with pytest.raises(ValueError, match="cannot be empty"):
            env_creds.set_env_key("TEST_KEY", "")

    def test_newline_in_value_raises(self, cfg_dir):
        with pytest.raises(ValueError, match="cannot contain newline"):
            env_creds.set_env_key("TEST_KEY", "line1\nline2")


class TestGetEnvKey:
    def test_returns_none_when_missing(self, cfg_dir):
        assert env_creds.get_env_key("MISSING_KEY") is None

    def test_returns_env_var(self, cfg_dir, monkeypatch):
        monkeypatch.setenv("ENV_ONLY", "from-env")
        assert env_creds.get_env_key("ENV_ONLY") == "from-env"

    def test_prefers_env_var_over_file(self, cfg_dir, monkeypatch):
        env_creds.set_env_key("DUPLICATE", "from-file")
        monkeypatch.setenv("DUPLICATE", "from-env")
        assert env_creds.get_env_key("DUPLICATE") == "from-env"

    def test_reads_export_syntax(self, cfg_dir):
        env_file = cfg_dir / "env"
        env_file.write_text("export FOO=bar\n", encoding="utf-8")
        assert env_creds.get_env_key("FOO") == "bar"

    def test_no_file_returns_none(self, cfg_dir):
        # cfg_dir has no env file yet
        assert env_creds.get_env_key("ANYTHING") is None

    def test_ignores_comments(self, cfg_dir):
        env_file = cfg_dir / "env"
        env_file.write_text("# COMMENTED=ignored\nREAL=value\n", encoding="utf-8")
        assert env_creds.get_env_key("COMMENTED") is None
        assert env_creds.get_env_key("REAL") == "value"


class TestOpenRouterDelegation:
    """Verify that openrouter_creds still works after refactoring."""

    def test_save_and_get_via_openrouter(self, cfg_dir, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from open_llm_proxy import openrouter_creds

        openrouter_creds.save_api_key("sk-or-test")
        assert openrouter_creds.get_api_key() == "sk-or-test"
        assert openrouter_creds.get_persisted_api_key() == "sk-or-test"

    def test_persisted_ignores_env(self, cfg_dir, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-shell")
        from open_llm_proxy import openrouter_creds

        with pytest.raises(RuntimeError, match="absent"):
            openrouter_creds.get_persisted_api_key()

    def test_get_api_key_falls_back_to_env_var(self, cfg_dir, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-shell")
        from open_llm_proxy import openrouter_creds

        assert openrouter_creds.get_api_key() == "sk-or-shell"


class TestEnvLineAccountAware:
    """Env-line providers (openrouter, nvidia) account-aware key readers."""

    def test_openrouter_named_account_resolves(self, cfg_dir, monkeypatch):
        """openrouter_creds.get_persisted_api_key(account='work') reads from
        per-account secret file."""
        from open_llm_proxy import account_registry

        account_registry.add_account(
            "openrouter",
            "work",
            storage="api-key",
            secret_bytes=b"sk-or-named-work",
        )
        from open_llm_proxy import openrouter_creds

        assert openrouter_creds.get_persisted_api_key(account="work") == "sk-or-named-work"

    def test_openrouter_default_still_reads_env_file(self, cfg_dir, monkeypatch):
        """Bare get_persisted_api_key() (no account) still reads from env file."""
        from open_llm_proxy import env_creds, openrouter_creds

        env_creds.set_env_key("OPENROUTER_API_KEY", "sk-or-env-file")
        assert openrouter_creds.get_persisted_api_key() == "sk-or-env-file"

    def test_nvidia_named_account_resolves(self, cfg_dir, monkeypatch):
        """nvidia_creds.get_api_key(account='work') reads from per-account file."""
        from open_llm_proxy import account_registry

        account_registry.add_account(
            "nvidia",
            "work",
            storage="api-key",
            secret_bytes=b"nv-named-work",
        )
        from open_llm_proxy import nvidia_creds

        assert nvidia_creds.get_api_key(account="work") == "nv-named-work"

    def test_nvidia_default_reads_env(self, cfg_dir, monkeypatch):
        """nvidia_creds.get_api_key() with no account reads from env / env file."""
        from open_llm_proxy import env_creds, nvidia_creds

        env_creds.set_env_key("NVIDIA_API_KEY", "nv-env-file")
        assert nvidia_creds.get_api_key() == "nv-env-file"

    # ---- HIGH — non-default account fails closed (no env-var fallback) ---------

    def test_openrouter_persisted_named_no_secret_raises(self, cfg_dir):
        """get_persisted_api_key(account='ghost') with no stored secret raises
        RuntimeError — does NOT fall back to env file."""
        from open_llm_proxy import openrouter_creds

        with pytest.raises(RuntimeError, match="has no stored credential"):
            openrouter_creds.get_persisted_api_key(account="ghost")

    def test_openrouter_get_api_key_named_no_secret_raises(self, cfg_dir, monkeypatch):
        """get_api_key(account='ghost') with no stored secret raises RuntimeError
        — does NOT fall back to env var."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
        from open_llm_proxy import openrouter_creds

        with pytest.raises(RuntimeError, match="has no stored credential"):
            openrouter_creds.get_api_key(account="ghost")

    def test_nvidia_named_no_secret_raises(self, cfg_dir, monkeypatch):
        """nvidia_creds.get_api_key(account='ghost') with no stored secret raises
        RuntimeError — does NOT fall back to env var."""
        monkeypatch.setenv("NVIDIA_API_KEY", "nv-env")
        from open_llm_proxy import nvidia_creds

        with pytest.raises(RuntimeError, match="has no stored credential"):
            nvidia_creds.get_api_key(account="ghost")

    def test_openrouter_persisted_named_default_legacy_fallback(self, cfg_dir):
        """get_persisted_api_key(account='default') without a file-backed
        'default' account still reads from the shared env file (legacy)."""
        from open_llm_proxy import env_creds, openrouter_creds

        env_creds.set_env_key("OPENROUTER_API_KEY", "sk-or-legacy-default")
        assert openrouter_creds.get_persisted_api_key(account="default") == "sk-or-legacy-default"

    def test_openrouter_get_api_key_named_default_env_fallback(self, cfg_dir, monkeypatch):
        """get_api_key(account='default') without a file-backed secret falls
        back to env var (legacy behavior preserved)."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-default")
        from open_llm_proxy import openrouter_creds

        assert openrouter_creds.get_api_key(account="default") == "sk-or-env-default"

    def test_nvidia_named_default_legacy_fallback(self, cfg_dir, monkeypatch):
        """nvidia_creds.get_api_key(account='default') without a file-backed
        secret falls back to env var (legacy behavior preserved)."""
        monkeypatch.setenv("NVIDIA_API_KEY", "nv-env-default")
        from open_llm_proxy import nvidia_creds

        assert nvidia_creds.get_api_key(account="default") == "nv-env-default"
