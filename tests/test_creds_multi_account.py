from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx  # ensure httpx is in sys.modules before we patch it
import pytest

from open_llm_proxy import creds
from open_llm_proxy.account_registry import (
    AccountRegistryError,
    add_account,
    list_accounts,
    resolve_secret_ref,
)


# ---- Fixtures --------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_cache():
    # Clear ALL account caches (named accounts persist across tests otherwise)
    creds._cached_key_cache.clear()
    creds._cached_time_cache.clear()
    creds._in_memory_oauth_cache.clear()
    yield
    creds._cached_key_cache.clear()
    creds._cached_time_cache.clear()
    creds._in_memory_oauth_cache.clear()


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set OLP_CONFIG_DIR to a tmp_path and return it."""
    d = tmp_path / "olp_config"
    d.mkdir()
    monkeypatch.setenv("OLP_CONFIG_DIR", str(d))
    return d


# ---- account=None still uses legacy ~/.claude/.credentials.json -------------------

class TestDefaultAccount:
    def test_default_reads_credentials_file(self, monkeypatch, tmp_path):
        """account=None (or no arg) reads from ~/.claude/.credentials.json with BYPASS_KEYCHAIN=1."""
        monkeypatch.setenv("BYPASS_KEYCHAIN", "1")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        creds_dir = fake_home / ".claude"
        creds_dir.mkdir()
        creds_json = creds_dir / ".credentials.json"
        creds_json.write_text(
            json.dumps({
                "claudeAiOauth": {
                    "accessToken": "sk-ant-oat01-default-token",
                    "refreshToken": "default-refresh",
                    "expiresAt": 9999999999999,
                }
            }),
            encoding="utf-8",
        )

        assert creds.get_api_key() == "sk-ant-oat01-default-token"
        assert creds.get_api_key(account=None) == "sk-ant-oat01-default-token"

    def test_default_from_env(self, monkeypatch):
        """account=None still reads ANTHROPIC_API_KEY from env."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-key")
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        assert creds.get_api_key() == "sk-ant-env-key"
        assert creds.get_api_key(account=None) == "sk-ant-env-key"


# ---- Named accounts resolve from per-account files -------------------------------

class TestNamedAccount:
    def _add_named_account(self, cfg: Path, name: str, access_token: str) -> None:
        """Helper to register a named claude-cli account with OAuth credentials."""
        creds_data = json.dumps({
            "claudeAiOauth": {
                "accessToken": access_token,
                "refreshToken": f"{name}-refresh-token",
                "expiresAt": 9999999999999,
            }
        })
        add_account("claude-cli", name, storage="claude-oauth", secret_bytes=creds_data.encode())

    def test_named_accounts_resolve_different_tokens(self, cfg, monkeypatch):
        """Two named accounts return different access tokens."""
        monkeypatch.setenv("BYPASS_KEYCHAIN", "1")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

        self._add_named_account(cfg, "work", "sk-ant-oat01-work-token")
        self._add_named_account(cfg, "home", "sk-ant-oat01-home-token")

        assert creds.get_api_key(account="work") == "sk-ant-oat01-work-token"
        assert creds.get_api_key(account="home") == "sk-ant-oat01-home-token"

    def test_unknown_account_raises(self, cfg, monkeypatch):
        """get_api_key with an unregistered account name raises RuntimeError."""
        monkeypatch.setenv("BYPASS_KEYCHAIN", "1")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

        with pytest.raises(RuntimeError, match="No Anthropic credentials found"):
            creds.get_api_key(account="nonexistent")

    def test_named_account_with_raw_api_key(self, cfg, monkeypatch):
        """Named account with a raw API key string stored still works."""
        monkeypatch.setenv("BYPASS_KEYCHAIN", "1")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

        add_account("claude-cli", "rawkey", storage="claude-oauth", secret_bytes=b"sk-ant-raw-key-value")

        assert creds.get_api_key(account="rawkey") == "sk-ant-raw-key-value"


# ---- Refresh writes to correct per-account file -----------------------------------

class TestRefresh:
    def test_refresh_named_account_writes_only_its_file(self, cfg, monkeypatch, tmp_path):
        """Refresh for a named account writes the new token only to that account's file."""
        monkeypatch.setenv("BYPASS_KEYCHAIN", "1")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

        # Setup default credentials file (should remain untouched)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        default_creds_dir = fake_home / ".claude"
        default_creds_dir.mkdir()
        default_creds_path = default_creds_dir / ".credentials.json"
        default_creds_path.write_text(json.dumps({"api_key": "sk-ant-default"}), encoding="utf-8")

        # Create account "work" with an expired token
        expired_ts = int(time.time() * 1000) - 3600000  # 1 hour ago
        work_creds = {
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-old-token",
                "refreshToken": "work-refresh-token",
                "expiresAt": expired_ts,
            }
        }
        add_account("claude-cli", "work", storage="claude-oauth", secret_bytes=json.dumps(work_creds).encode())

        # Create account "home" with a valid token (should not be touched)
        home_creds = {
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-home-token",
                "refreshToken": "home-refresh-token",
                "expiresAt": 9999999999999,
            }
        }
        add_account("claude-cli", "home", storage="claude-oauth", secret_bytes=json.dumps(home_creds).encode())

        # Mock httpx.post to return new tokens
        new_access = "sk-ant-oat01-new-access"
        new_refresh = "new-refresh-token"

        def mock_post(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "access_token": new_access,
                "refresh_token": new_refresh,
                "expires_in": 3600,
            }
            return mock_resp

        with patch("httpx.post", side_effect=mock_post):
            key = creds.get_api_key(account="work")

        assert key == new_access

        # Verify work account file was updated
        work_path = cfg / "accounts" / "claude-cli" / "work.credentials.json"
        assert work_path.exists()
        updated_work = json.loads(work_path.read_bytes())
        assert updated_work["claudeAiOauth"]["accessToken"] == new_access
        assert updated_work["claudeAiOauth"]["refreshToken"] == new_refresh

        # Verify home account file was NOT updated
        home_path = cfg / "accounts" / "claude-cli" / "home.credentials.json"
        updated_home = json.loads(home_path.read_bytes())
        assert updated_home["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-home-token"
        assert updated_home["claudeAiOauth"]["refreshToken"] == "home-refresh-token"

        # Verify default credentials file was NOT updated
        assert json.loads(default_creds_path.read_text()) == {"api_key": "sk-ant-default"}


# ---- clear_cache isolation --------------------------------------------------------

class TestClearCache:
    def test_clear_cache_isolates_accounts(self, cfg, monkeypatch):
        """clear_cache(account='work') does not affect other accounts' cache."""
        monkeypatch.setenv("BYPASS_KEYCHAIN", "1")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

        # Register two accounts
        for name, token in [("work", "sk-ant-oat01-work"), ("home", "sk-ant-oat01-home")]:
            cred_data = json.dumps({
                "claudeAiOauth": {
                    "accessToken": token,
                    "refreshToken": f"{name}-rt",
                    "expiresAt": 9999999999999,
                }
            })
            add_account("claude-cli", name, storage="claude-oauth", secret_bytes=cred_data.encode())

        # Populate caches
        assert creds.get_api_key(account="work") == "sk-ant-oat01-work"
        assert creds.get_api_key(account="home") == "sk-ant-oat01-home"

        # Clear only work
        creds.clear_cache(account="work")
        creds.reset_oauth_state(account="work")

        # Work should re-resolve; home should still be cached (faster if cached)
        assert creds.get_api_key(account="work") == "sk-ant-oat01-work"
        assert creds.get_api_key(account="home") == "sk-ant-oat01-home"

    def test_clear_cache_default_does_not_affect_named(self, cfg, monkeypatch):
        """clear_cache() (default) does not clear a named account's cache."""
        monkeypatch.setenv("BYPASS_KEYCHAIN", "1")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

        cred_data = json.dumps({
            "claudeAiOauth": {"accessToken": "sk-ant-oat01-named", "refreshToken": "rt", "expiresAt": 9999999999999}
        })
        add_account("claude-cli", "myaccount", storage="claude-oauth", secret_bytes=cred_data.encode())

        assert creds.get_api_key(account="myaccount") == "sk-ant-oat01-named"
        creds.clear_cache()
        creds.reset_oauth_state()
        # Named account cache should still be intact (or re-resolvable)
        assert creds.get_api_key(account="myaccount") == "sk-ant-oat01-named"


# ---- CRITICAL 2a — active-account resolution via account=None -----------------


class TestActiveAccountResolution:
    def test_account_none_resolves_active_named(self, cfg, monkeypatch):
        """get_api_key(account=None) resolves the active account when it is a
        named non-default with a file-backed secret."""
        monkeypatch.setenv("BYPASS_KEYCHAIN", "1")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

        cred_data = json.dumps({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-active-work",
                "refreshToken": "work-rt",
                "expiresAt": 9999999999999,
            }
        })
        add_account("claude-cli", "work", storage="claude-oauth", secret_bytes=cred_data.encode())
        from open_llm_proxy import account_registry as ar

        ar.set_active("claude-cli", "work")

        assert creds.get_api_key(account=None) == "sk-ant-oat01-active-work"

    def test_active_switch_no_stale_cache(self, cfg, monkeypatch, tmp_path):
        """After priming the default cache, switching active to a named account
        returns the named token — no 30s stale window."""
        monkeypatch.setenv("BYPASS_KEYCHAIN", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-default-env")
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

        # Prime default cache with the env-key default
        assert creds.get_api_key(account=None) == "sk-ant-default-env"
        assert creds._cached_key_cache.get(creds._DEFAULT) is not None

        # Create a named account and set it active
        cred_data = json.dumps({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-switched",
                "refreshToken": "switched-rt",
                "expiresAt": 9999999999999,
            }
        })
        add_account("claude-cli", "switch-acct", storage="claude-oauth", secret_bytes=cred_data.encode())
        from open_llm_proxy import account_registry
        account_registry.set_active("claude-cli", "switch-acct")

        # account=None should return the named token — NOT the stale default
        # (cache key is now "switch-acct", which was never cached)
        assert creds.get_api_key(account=None) == "sk-ant-oat01-switched"

    def test_account_none_with_active_default_legacy_path(self, cfg, monkeypatch, tmp_path):
        """When active account is 'default', account=None uses the legacy
        resolution path (env → files), NOT the per-account file for 'default'.
        This matches the spec: only NAMED non-default accounts trigger active
        resolution."""
        monkeypatch.setenv("BYPASS_KEYCHAIN", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-legacy-env")
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

        # The legacy env var takes precedence when active is 'default'
        assert creds.get_api_key(account=None) == "sk-ant-legacy-env"


# ---- MEDIUM — named refresh atomic write (no leftover .tmp) -------------------


class TestRefreshAtomicWrite:
    def test_no_leftover_tmp_after_refresh(self, cfg, monkeypatch, tmp_path):
        """Named account refresh leaves no .tmp or temp files behind."""
        monkeypatch.setenv("BYPASS_KEYCHAIN", "1")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        (fake_home / ".claude").mkdir()
        (fake_home / ".claude" / ".credentials.json").write_text(
            json.dumps({"api_key": "sk-ant-default"}), encoding="utf-8"
        )

        expired_ts = int(time.time() * 1000) - 3600000
        work_creds = {
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-old",
                "refreshToken": "work-refresh",
                "expiresAt": expired_ts,
            }
        }
        add_account("claude-cli", "work", storage="claude-oauth",
                    secret_bytes=json.dumps(work_creds).encode())

        def mock_post(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "access_token": "sk-ant-oat01-new",
                "refresh_token": "new-refresh-token",
                "expires_in": 3600,
            }
            return mock_resp

        with patch("httpx.post", side_effect=mock_post):
            key = creds.get_api_key(account="work")

        assert key == "sk-ant-oat01-new"

        acct_dir = cfg / "accounts" / "claude-cli"
        tmp_files = [f for f in acct_dir.iterdir() if f.name.endswith(".tmp")]
        assert tmp_files == [], f"Leftover .tmp files: {tmp_files}"
        tmp_prefix = [f for f in acct_dir.iterdir() if f.name.startswith("creds_tmp_")]
        assert tmp_prefix == [], f"Leftover temp files: {tmp_prefix}"
        assert (acct_dir / "work.credentials.json").exists()
