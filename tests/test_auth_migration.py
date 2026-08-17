from __future__ import annotations

from pathlib import Path

import pytest

# ---- Helpers ------------------------------------------------------------------


def stub_all_creds_found(monkeypatch):
    """Make all legacy credential getters return a value."""
    import open_llm_proxy.copilot_creds
    import open_llm_proxy.creds
    import open_llm_proxy.nvidia_creds
    import open_llm_proxy.opencode_creds
    import open_llm_proxy.openrouter_creds

    monkeypatch.setattr(
        open_llm_proxy.openrouter_creds,
        "get_persisted_api_key",
        lambda: "sk-or-migrated",
    )
    monkeypatch.setattr(
        open_llm_proxy.opencode_creds,
        "get_opencode_api_key",
        lambda: "oc-key-migrated",
    )
    monkeypatch.setattr(
        open_llm_proxy.copilot_creds,
        "get_oauth_token",
        lambda: "gho_migrated",
    )
    monkeypatch.setattr(
        open_llm_proxy.creds,
        "get_api_key",
        lambda: "sk-ant-migrated",
    )
    monkeypatch.setattr(
        open_llm_proxy.nvidia_creds,
        "get_api_key",
        lambda: "nv-migrated",
    )


def stub_all_creds_missing(monkeypatch):
    """Make all legacy credential getters raise so migration is no-op."""
    import open_llm_proxy.copilot_creds
    import open_llm_proxy.creds
    import open_llm_proxy.nvidia_creds
    import open_llm_proxy.opencode_creds
    import open_llm_proxy.openrouter_creds

    def _raise(msg: str):
        def _inner():
            raise RuntimeError(msg)

        return _inner

    monkeypatch.setattr(
        open_llm_proxy.openrouter_creds,
        "get_persisted_api_key",
        _raise("no openrouter cred"),
    )
    monkeypatch.setattr(
        open_llm_proxy.opencode_creds,
        "get_opencode_api_key",
        _raise("no opencode cred"),
    )
    monkeypatch.setattr(
        open_llm_proxy.copilot_creds,
        "get_oauth_token",
        _raise("no copilot cred"),
    )
    monkeypatch.setattr(
        open_llm_proxy.creds,
        "get_api_key",
        _raise("no claude cred"),
    )
    monkeypatch.setattr(
        open_llm_proxy.nvidia_creds,
        "get_api_key",
        _raise("no nvidia cred"),
    )


# ---- Fixtures -----------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set OLP_CONFIG_DIR to a temp directory."""
    d = tmp_path / "olp_config"
    d.mkdir()
    monkeypatch.setenv("OLP_CONFIG_DIR", str(d))
    return d


# ---- Tests --------------------------------------------------------------------


class TestMigrateLegacyCredentials:
    def test_migrates_found_credentials(self, cfg, monkeypatch):
        stub_all_creds_found(monkeypatch)
        from open_llm_proxy.auth_migration import migrate_legacy_credentials

        migrated = migrate_legacy_credentials()
        assert sorted(migrated) == [
            "claude-cli",
            "github-copilot",
            "nvidia",
            "opencode",
            "openrouter",
        ]

        # Verify each was registered as @default
        from open_llm_proxy import account_registry

        for prov in ("openrouter", "opencode", "github-copilot", "claude-cli", "nvidia"):
            accounts = account_registry.list_accounts(prov)
            assert len(accounts) == 1
            assert accounts[0].name == "default"
            assert accounts[0].is_active is True

    def test_openrouter_env_line_ref(self, cfg, monkeypatch):
        stub_all_creds_found(monkeypatch)
        from open_llm_proxy.auth_migration import migrate_legacy_credentials

        migrate_legacy_credentials()
        from open_llm_proxy import account_registry

        info = account_registry.list_accounts("openrouter")[0]
        assert info.storage == "env-line"
        assert info.ref == "OPENROUTER_API_KEY"

    def test_opencode_external_ref(self, cfg, monkeypatch):
        stub_all_creds_found(monkeypatch)
        from open_llm_proxy.auth_migration import migrate_legacy_credentials

        migrate_legacy_credentials()
        from open_llm_proxy import account_registry

        info = account_registry.list_accounts("opencode")[0]
        assert info.storage == "external"
        assert info.ref == "opencode"

    def test_github_copilot_external_ref(self, cfg, monkeypatch):
        stub_all_creds_found(monkeypatch)
        from open_llm_proxy.auth_migration import migrate_legacy_credentials

        migrate_legacy_credentials()
        from open_llm_proxy import account_registry

        info = account_registry.list_accounts("github-copilot")[0]
        assert info.storage == "external"
        assert info.ref == "copilot"

    def test_claude_cli_external_ref(self, cfg, monkeypatch):
        stub_all_creds_found(monkeypatch)
        from open_llm_proxy.auth_migration import migrate_legacy_credentials

        migrate_legacy_credentials()
        from open_llm_proxy import account_registry

        info = account_registry.list_accounts("claude-cli")[0]
        assert info.storage == "external"
        assert info.ref == "claude-default"

    def test_idempotent_second_call_noop(self, cfg, monkeypatch):
        stub_all_creds_found(monkeypatch)
        from open_llm_proxy.auth_migration import migrate_legacy_credentials

        migrate_legacy_credentials()
        second = migrate_legacy_credentials()
        assert second == []

        from open_llm_proxy import account_registry

        for prov in ("openrouter", "opencode", "github-copilot", "claude-cli", "nvidia"):
            assert len(account_registry.list_accounts(prov)) == 1

    def test_skips_provider_that_raises(self, cfg, monkeypatch):
        import open_llm_proxy.openrouter_creds

        # Only openrouter works; opencode, copilot, claude, nvidia raise
        stub_all_creds_missing(monkeypatch)
        monkeypatch.setattr(
            open_llm_proxy.openrouter_creds,
            "get_persisted_api_key",
            lambda: "sk-or-only",
        )
        from open_llm_proxy.auth_migration import migrate_legacy_credentials

        migrated = migrate_legacy_credentials()
        assert migrated == ["openrouter"]

        from open_llm_proxy import account_registry

        assert len(account_registry.list_accounts("openrouter")) == 1
        assert account_registry.list_accounts("opencode") == []
        assert account_registry.list_accounts("github-copilot") == []
        assert account_registry.list_accounts("claude-cli") == []
        assert account_registry.list_accounts("nvidia") == []

    def test_no_credentials_does_nothing(self, cfg, monkeypatch):
        stub_all_creds_missing(monkeypatch)
        from open_llm_proxy.auth_migration import migrate_legacy_credentials

        migrated = migrate_legacy_credentials()
        assert migrated == []

        from open_llm_proxy import account_registry

        assert account_registry.list_providers() == []
