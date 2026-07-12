from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest

from open_llm_proxy.account_registry import (
    CONFIG_DIR,
    AccountInfo,
    AccountRegistryError,
    active_account,
    add_account,
    list_accounts,
    list_providers,
    load,
    normalize_account_name,
    read_secret,
    remove_account,
    rename_account,
    resolve_secret_ref,
    set_active,
)


# ---- Fixtures --------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set OLP_CONFIG_DIR to a tmp_path and return it."""
    d = tmp_path / "olp_config"
    d.mkdir()
    monkeypatch.setenv("OLP_CONFIG_DIR", str(d))
    return d


# ---- normalize_account_name --------------------------------------------------------

class TestNormalizeAccountName:
    def test_lowercases(self):
        assert normalize_account_name("MyAccount") == "myaccount"

    def test_allows_underscore_dash(self):
        assert normalize_account_name("my-work_account") == "my-work_account"

    def test_empty_raises(self):
        with pytest.raises(AccountRegistryError, match="not be empty"):
            normalize_account_name("")

    def test_too_long_raises(self):
        with pytest.raises(AccountRegistryError, match="at most 32"):
            normalize_account_name("a" * 33)

    def test_leading_dash_raises(self):
        with pytest.raises(AccountRegistryError, match="Invalid account name"):
            normalize_account_name("-bad")

    def test_uppercase_is_lowercased(self):
        assert normalize_account_name("UPPERCASE") == "uppercase"

    def test_special_chars_raises(self):
        with pytest.raises(AccountRegistryError, match="Invalid account name"):
            normalize_account_name("bad name!")

    def test_starts_with_digit_ok(self):
        assert normalize_account_name("1account") == "1account"


# ---- add_account ------------------------------------------------------------------

class TestAddAccount:
    def test_add_first_auto_names_default_and_active(self, cfg: Path):
        info = add_account("claude-cli", storage="api-key", secret_bytes=b"sk-123")
        assert info.name == "default"
        assert info.is_active is True
        assert info.provider == "claude-cli"
        assert info.storage == "api-key"
        assert info.ref.startswith("accounts/")

        accounts = list_accounts("claude-cli")
        assert len(accounts) == 1
        assert accounts[0].name == "default"
        assert accounts[0].is_active is True

    def test_add_second_with_explicit_name(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk-1")
        info2 = add_account("claude-cli", "work", storage="api-key", secret_bytes=b"sk-2")

        assert info2.name == "work"
        assert info2.is_active is False

        accounts = list_accounts("claude-cli")
        assert len(accounts) == 2
        names = {a.name for a in accounts}
        assert names == {"default", "work"}
        default_info = next(a for a in accounts if a.name == "default")
        assert default_info.is_active is True

    def test_add_second_with_name_none_raises(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk-1")
        with pytest.raises(AccountRegistryError, match="name required"):
            add_account("claude-cli", storage="api-key", secret_bytes=b"sk-2")

    def test_duplicate_name_raises(self, cfg: Path):
        add_account("claude-cli", "mine", storage="api-key", secret_bytes=b"sk-1")
        with pytest.raises(AccountRegistryError, match="already exists"):
            add_account("claude-cli", "mine", storage="api-key", secret_bytes=b"sk-2")

    def test_invalid_name_through_add_raises(self, cfg: Path):
        with pytest.raises(AccountRegistryError, match="Invalid account name"):
            add_account("claude-cli", "bad name!", storage="api-key", secret_bytes=b"x")

    def test_neither_secret_nor_ref_raises(self, cfg: Path):
        with pytest.raises(AccountRegistryError, match="secret_bytes or ref"):
            add_account("openrouter", storage="env-line")

    def test_env_line_with_ref_no_file_created(self, cfg: Path):
        info = add_account(
            "openrouter", storage="env-line", ref="OPENROUTER_API_KEY"
        )
        assert info.ref == "OPENROUTER_API_KEY"
        # verify no secret file was created
        assert read_secret("openrouter", "default") is None
        # registry has exactly that ref
        raw = load()
        assert raw["providers"]["openrouter"]["accounts"]["default"]["ref"] == "OPENROUTER_API_KEY"


# ---- list_accounts / list_providers / active_account / load -----------------------

class TestQuery:
    def test_list_providers_sorted(self, cfg: Path):
        add_account("z-provider", storage="api-key", secret_bytes=b"x")
        add_account("a-provider", storage="api-key", secret_bytes=b"y")
        assert list_providers() == ["a-provider", "z-provider"]

    def test_list_providers_empty_when_no_accounts(self, cfg: Path):
        assert list_providers() == []

    def test_list_accounts_unknown_provider_returns_empty(self, cfg: Path):
        assert list_accounts("nonexistent") == []

    def test_active_account_returns_none_for_unknown(self, cfg: Path):
        assert active_account("nonexistent") is None

    def test_active_account_after_add(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk")
        assert active_account("claude-cli") == "default"

    def test_load_empty(self, cfg: Path):
        assert load() == {"version": 1, "providers": {}}

    def test_load_after_add(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk")
        raw = load()
        assert raw["version"] == 1
        assert "claude-cli" in raw["providers"]


# ---- rename_account ----------------------------------------------------------------

class TestRenameAccount:
    def test_rename_guard_single_account_raises(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk")
        with pytest.raises(AccountRegistryError, match="at least 2 accounts"):
            rename_account("claude-cli", "default", "primary")

    def test_rename_with_two_accounts_succeeds(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk-1")
        add_account("claude-cli", "work", storage="api-key", secret_bytes=b"sk-2")

        rename_account("claude-cli", "default", "primary")

        accounts = list_accounts("claude-cli")
        names = {a.name for a in accounts}
        assert names == {"primary", "work"}
        # active pointer moved
        assert active_account("claude-cli") == "primary"

    def test_rename_moves_secret_file(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk-1")
        add_account("claude-cli", "work", storage="api-key", secret_bytes=b"sk-2")

        old_ref = list_accounts("claude-cli")[0].ref  # default's ref
        rename_account("claude-cli", "default", "primary")

        primary = next(a for a in list_accounts("claude-cli") if a.name == "primary")
        assert primary.ref != old_ref
        assert "primary" in primary.ref
        old_path = cfg / old_ref
        new_path = cfg / primary.ref
        assert not old_path.exists()
        assert new_path.exists()
        assert new_path.read_bytes() == b"sk-1"

    def test_rename_unknown_provider_raises(self, cfg: Path):
        with pytest.raises(AccountRegistryError, match="Unknown provider"):
            rename_account("ghost", "old", "new")

    def test_rename_nonexistent_account_raises(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk")
        add_account("claude-cli", "work", storage="api-key", secret_bytes=b"sk2")
        with pytest.raises(AccountRegistryError, match="not found"):
            rename_account("claude-cli", "nope", "new")

    def test_rename_to_existing_raises(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk")
        add_account("claude-cli", "work", storage="api-key", secret_bytes=b"sk2")
        with pytest.raises(AccountRegistryError, match="already exists"):
            rename_account("claude-cli", "default", "work")


# ---- set_active -------------------------------------------------------------------

class TestSetActive:
    def test_set_active_switches(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk-1")
        add_account("claude-cli", "work", storage="api-key", secret_bytes=b"sk-2")

        set_active("claude-cli", "work")
        assert active_account("claude-cli") == "work"

    def test_set_active_unknown_raises(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk")
        with pytest.raises(AccountRegistryError, match="not found"):
            set_active("claude-cli", "nope")

    def test_set_active_unknown_provider_raises(self, cfg: Path):
        with pytest.raises(AccountRegistryError, match="Unknown provider"):
            set_active("ghost", "default")


# ---- remove_account ---------------------------------------------------------------

class TestRemoveAccount:
    def test_remove_non_last_repoints_active(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk-1")
        add_account("claude-cli", "work", storage="api-key", secret_bytes=b"sk-2")

        remove_account("claude-cli", "work")
        accounts = list_accounts("claude-cli")
        assert len(accounts) == 1
        assert accounts[0].name == "default"
        assert active_account("claude-cli") == "default"

    def test_remove_non_last_removes_secret_file(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk-1")
        add_account("claude-cli", "work", storage="api-key", secret_bytes=b"sk-2")

        work_ref = next(a.ref for a in list_accounts("claude-cli") if a.name == "work")
        work_path = cfg / work_ref
        assert work_path.exists()

        remove_account("claude-cli", "work")
        assert not work_path.exists()

    def test_remove_last_without_force_raises(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk")
        with pytest.raises(AccountRegistryError, match="Cannot remove last account"):
            remove_account("claude-cli", "default")

    def test_remove_last_with_force_drops_provider(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk")
        remove_account("claude-cli", "default", force=True)
        assert list_providers() == []
        assert list_accounts("claude-cli") == []

    def test_remove_active_repoints(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk-1")
        add_account("claude-cli", "work", storage="api-key", secret_bytes=b"sk-2")
        set_active("claude-cli", "work")

        remove_account("claude-cli", "work")
        assert active_account("claude-cli") == "default"

    def test_remove_unknown_provider_raises(self, cfg: Path):
        with pytest.raises(AccountRegistryError, match="Unknown provider"):
            remove_account("ghost", "default")

    def test_remove_nonexistent_account_raises(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk")
        with pytest.raises(AccountRegistryError, match="not found"):
            remove_account("claude-cli", "nope")


# ---- Secret file round-trip & permissions -----------------------------------------

class TestSecrets:
    def test_read_secret_round_trip(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"my-secret-key")
        assert read_secret("claude-cli", "default") == b"my-secret-key"

    def test_read_secret_unknown_provider_raises(self, cfg: Path):
        with pytest.raises(AccountRegistryError, match="Unknown provider"):
            read_secret("ghost", "default")

    def test_read_secret_unknown_account_raises(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk")
        with pytest.raises(AccountRegistryError, match="not found"):
            read_secret("claude-cli", "nope")

    def test_secret_file_0600_perms(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"secret")
        info = list_accounts("claude-cli")[0]
        fpath = cfg / info.ref
        assert fpath.exists()
        mode = fpath.stat().st_mode
        assert mode & 0o777 == 0o600, hex(mode)

    def test_registry_file_0600_perms(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk")
        reg = cfg / "accounts.json"
        assert reg.exists()
        mode = reg.stat().st_mode
        assert mode & 0o777 == 0o600, hex(mode)


# ---- resolve_secret_ref -----------------------------------------------------------

class TestResolveSecretRef:
    def test_file_ref_returns_absolute_path(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk")
        resolved = resolve_secret_ref("claude-cli", "default")
        assert isinstance(resolved, Path)
        assert resolved.is_absolute()
        assert "accounts/claude-cli/default.key" in str(resolved)

    def test_env_var_ref_returns_string(self, cfg: Path):
        add_account("openrouter", storage="env-line", ref="OPENROUTER_API_KEY")
        resolved = resolve_secret_ref("openrouter", "default")
        assert isinstance(resolved, str)
        assert resolved == "OPENROUTER_API_KEY"

    def test_unknown_provider_raises(self, cfg: Path):
        with pytest.raises(AccountRegistryError, match="Unknown provider"):
            resolve_secret_ref("ghost", "default")

    def test_unknown_account_raises(self, cfg: Path):
        add_account("claude-cli", storage="api-key", secret_bytes=b"sk")
        with pytest.raises(AccountRegistryError, match="not found"):
            resolve_secret_ref("claude-cli", "nope")


# ---- LOW — chmod failure raises AccountRegistryError ----------------------------


class TestChmodFailure:
    def test_chmod_failure_raises_registry_error(self, cfg, monkeypatch):
        """When os.chmod fails on the config directory, AccountRegistryError
        is raised with a safe (non-secret) message."""
        import open_llm_proxy.account_registry as ar

        def bad_chmod(*args, **kwargs):
            raise PermissionError("chmod failed")

        monkeypatch.setattr(os, "chmod", bad_chmod)

        # The registry write goes through os.chmod twice (dir + file).
        # The first chmod on the parent dir will trigger an AccountRegistryError
        # because the config dir exists but chmod fails.
        with pytest.raises(AccountRegistryError, match="Failed to set permissions"):
            ar._write_registry({"version": 1, "providers": {}})

    def test_secret_file_chmod_failure_raises(self, cfg, monkeypatch):
        """When os.chmod fails during _write_secret_file, AccountRegistryError
        is raised."""
        import open_llm_proxy.account_registry as ar

        def bad_chmod(*args, **kwargs):
            raise PermissionError("chmod failed")

        monkeypatch.setattr(os, "chmod", bad_chmod)

        with pytest.raises(AccountRegistryError, match="Failed to set permissions"):
            ar._write_secret_file(cfg / "secret.key", b"secret-data")
