from __future__ import annotations

import io
from pathlib import Path

import pytest

from open_llm_proxy import cli

# ---- Fixtures -----------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set OLP_CONFIG_DIR to a temp directory."""
    d = tmp_path / "olp_config"
    d.mkdir()
    monkeypatch.setenv("OLP_CONFIG_DIR", str(d))
    return d


def _stub_migration_skipped(monkeypatch):
    """Make all legacy credential getters raise so migration is a no-op."""
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
        _raise("no cred"),
    )
    monkeypatch.setattr(
        open_llm_proxy.opencode_creds,
        "get_opencode_api_key",
        _raise("no cred"),
    )
    monkeypatch.setattr(
        open_llm_proxy.copilot_creds,
        "get_oauth_token",
        _raise("no cred"),
    )
    monkeypatch.setattr(
        open_llm_proxy.creds,
        "get_api_key",
        _raise("no cred"),
    )
    monkeypatch.setattr(
        open_llm_proxy.nvidia_creds,
        "get_api_key",
        _raise("no cred"),
    )


def _stub_migration_found_all(monkeypatch):
    """Make all legacy credential getters return a value so migration imports
    every provider as @default."""
    import open_llm_proxy.copilot_creds
    import open_llm_proxy.creds
    import open_llm_proxy.nvidia_creds
    import open_llm_proxy.opencode_creds
    import open_llm_proxy.openrouter_creds

    monkeypatch.setattr(
        open_llm_proxy.openrouter_creds,
        "get_persisted_api_key",
        lambda: "sk-or-test",
    )
    monkeypatch.setattr(
        open_llm_proxy.opencode_creds,
        "get_opencode_api_key",
        lambda: "oc-key-test",
    )
    monkeypatch.setattr(
        open_llm_proxy.copilot_creds,
        "get_oauth_token",
        lambda: "gho_test",
    )
    monkeypatch.setattr(
        open_llm_proxy.creds,
        "get_api_key",
        lambda: "sk-ant-test",
    )
    monkeypatch.setattr(
        open_llm_proxy.nvidia_creds,
        "get_api_key",
        lambda: "nv-test",
    )


def _openrouter_with_stdin(monkeypatch, text: str):
    """Patch stdin so openrouter key capture reads *text*."""
    import open_llm_proxy.openrouter_creds

    # Stub the creds getter used by migration
    monkeypatch.setattr(
        open_llm_proxy.openrouter_creds,
        "get_persisted_api_key",
        lambda: (_ for _ in ()).throw(RuntimeError("no cred")),
    )
    # Stub save_api_key to be a no-op in tests
    monkeypatch.setattr(
        open_llm_proxy.openrouter_creds,
        "save_api_key",
        lambda key: None,
    )
    # Fake stdin with the key
    fake_stdin = io.StringIO(text)
    monkeypatch.setattr(cli.sys, "stdin", fake_stdin)
    monkeypatch.setattr(fake_stdin, "isatty", lambda: False)


def _openrouter_with_tty(monkeypatch, key: str):
    """Patch getpass so openrouter key capture returns *key* interactively."""
    import open_llm_proxy.openrouter_creds

    monkeypatch.setattr(
        open_llm_proxy.openrouter_creds,
        "get_persisted_api_key",
        lambda: (_ for _ in ()).throw(RuntimeError("no cred")),
    )
    monkeypatch.setattr(
        open_llm_proxy.openrouter_creds,
        "save_api_key",
        lambda key: None,
    )
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("getpass.getpass", lambda prompt: key)


# ---- auth accounts ------------------------------------------------------------


class TestAuthAccounts:
    def test_no_provider_lists_all_migrated(self, cfg, monkeypatch, capsys):
        _stub_migration_found_all(monkeypatch)
        assert cli.main(["auth", "accounts"]) == 0
        out = capsys.readouterr().out
        assert "openrouter:" in out
        assert "opencode:" in out
        assert "github-copilot:" in out
        assert "claude-cli:" in out
        assert "nvidia:" in out
        # Each has an active default
        assert out.count("* default") == 5

    def test_specific_provider(self, cfg, monkeypatch, capsys):
        _stub_migration_found_all(monkeypatch)
        assert cli.main(["auth", "accounts", "openrouter"]) == 0
        out = capsys.readouterr().out
        assert "openrouter:" in out
        assert "* default" in out
        assert "opencode:" not in out

    def test_empty_when_no_credentials(self, cfg, monkeypatch, capsys):
        _stub_migration_skipped(monkeypatch)
        assert cli.main(["auth", "accounts"]) == 0
        out = capsys.readouterr().out
        assert "No credentials configured" in out

    def test_unknown_provider_errors(self, cfg, capsys):
        # auth accounts <provider> uses choices=KNOWN_PROVIDERS, so argparse
        # rejects unknown values before our handler runs (calls sys.exit(2)).
        with pytest.raises(SystemExit):
            cli.main(["auth", "accounts", "nonexistent"])

    def test_multi_account_listing(self, cfg, monkeypatch, capsys):
        _openrouter_with_stdin(monkeypatch, "sk-or-1\n")
        cli.main(["auth", "add", "openrouter"])
        capsys.readouterr()

        _openrouter_with_stdin(monkeypatch, "sk-or-2\n")
        cli.main(["auth", "add", "openrouter", "--name", "work"])
        capsys.readouterr()

        assert cli.main(["auth", "accounts", "openrouter"]) == 0
        out = capsys.readouterr().out
        assert "  * default" in out
        assert "    work" in out


# ---- auth add -----------------------------------------------------------------


class TestAuthAdd:
    def test_add_openrouter_auto_default(self, cfg, monkeypatch, capsys):
        _openrouter_with_stdin(monkeypatch, "sk-or-abc\n")
        assert cli.main(["auth", "add", "openrouter"]) == 0
        assert "default" in capsys.readouterr().out

        from open_llm_proxy import account_registry

        accounts = account_registry.list_accounts("openrouter")
        assert len(accounts) == 1
        assert accounts[0].name == "default"
        assert accounts[0].is_active is True

    def test_add_openrouter_second_named(self, cfg, monkeypatch, capsys):
        _openrouter_with_stdin(monkeypatch, "sk-or-1\n")
        cli.main(["auth", "add", "openrouter"])
        capsys.readouterr()

        _openrouter_with_stdin(monkeypatch, "sk-or-2\n")
        assert cli.main(["auth", "add", "openrouter", "--name", "work"]) == 0
        assert "work" in capsys.readouterr().out

        from open_llm_proxy import account_registry

        accounts = account_registry.list_accounts("openrouter")
        assert len(accounts) == 2
        names = {a.name for a in accounts}
        assert names == {"default", "work"}

    def test_add_openrouter_second_needs_name(self, cfg, monkeypatch, capsys):
        _openrouter_with_stdin(monkeypatch, "sk-or-1\n")
        cli.main(["auth", "add", "openrouter"])
        capsys.readouterr()

        _openrouter_with_stdin(monkeypatch, "sk-or-2\n")
        assert cli.main(["auth", "add", "openrouter"]) == 1
        err = capsys.readouterr().err
        assert "use --name" in err

    def test_add_openrouter_empty_key(self, cfg, monkeypatch, capsys):
        # Empty piped key should fail
        import open_llm_proxy.openrouter_creds

        monkeypatch.setattr(
            open_llm_proxy.openrouter_creds,
            "get_persisted_api_key",
            lambda: (_ for _ in ()).throw(RuntimeError("no cred")),
        )
        fake_stdin = io.StringIO("   \n")
        monkeypatch.setattr(cli.sys, "stdin", fake_stdin)
        monkeypatch.setattr(fake_stdin, "isatty", lambda: False)

        assert cli.main(["auth", "add", "openrouter"]) == 1
        err = capsys.readouterr().err
        assert "API key cannot be empty" in err


# ---- auth rename --------------------------------------------------------------


class TestAuthRename:
    def test_rename_single_account_errors_friendly(self, cfg, monkeypatch, capsys):
        _openrouter_with_stdin(monkeypatch, "sk-or-1\n")
        cli.main(["auth", "add", "openrouter"])
        capsys.readouterr()

        assert cli.main(["auth", "rename", "openrouter", "default", "primary"]) == 1
        err = capsys.readouterr().err
        assert "at least 2 accounts" in err

    def test_rename_with_two_accounts_succeeds(self, cfg, monkeypatch, capsys):
        _openrouter_with_stdin(monkeypatch, "sk-or-1\n")
        cli.main(["auth", "add", "openrouter"])
        capsys.readouterr()
        _openrouter_with_stdin(monkeypatch, "sk-or-2\n")
        cli.main(["auth", "add", "openrouter", "--name", "work"])
        capsys.readouterr()

        assert cli.main(["auth", "rename", "openrouter", "default", "primary"]) == 0
        assert "renamed" in capsys.readouterr().out.lower()

        from open_llm_proxy import account_registry

        accounts = account_registry.list_accounts("openrouter")
        names = {a.name for a in accounts}
        assert names == {"primary", "work"}
        assert account_registry.active_account("openrouter") == "primary"

    def test_rename_nonexistent(self, cfg, monkeypatch, capsys):
        _openrouter_with_stdin(monkeypatch, "sk-or-1\n")
        cli.main(["auth", "add", "openrouter"])
        capsys.readouterr()
        _openrouter_with_stdin(monkeypatch, "sk-or-2\n")
        cli.main(["auth", "add", "openrouter", "--name", "work"])
        capsys.readouterr()

        assert cli.main(["auth", "rename", "openrouter", "nope", "new"]) == 1
        assert "not found" in capsys.readouterr().err


# ---- auth use -----------------------------------------------------------------


class TestAuthUse:
    def test_use_changes_active(self, cfg, monkeypatch, capsys):
        _openrouter_with_stdin(monkeypatch, "sk-or-1\n")
        cli.main(["auth", "add", "openrouter"])
        capsys.readouterr()
        _openrouter_with_stdin(monkeypatch, "sk-or-2\n")
        cli.main(["auth", "add", "openrouter", "--name", "work"])
        capsys.readouterr()

        assert cli.main(["auth", "use", "openrouter", "work"]) == 0
        from open_llm_proxy import account_registry

        assert account_registry.active_account("openrouter") == "work"

    def test_use_nonexistent(self, cfg, monkeypatch, capsys):
        _openrouter_with_stdin(monkeypatch, "sk-or-1\n")
        cli.main(["auth", "add", "openrouter"])
        capsys.readouterr()

        assert cli.main(["auth", "use", "openrouter", "nope"]) == 1
        assert "not found" in capsys.readouterr().err

    def test_use_invalidates_creds_cache(self, cfg, monkeypatch, capsys):
        """auth use clears the credential cache (defense in depth)."""
        from open_llm_proxy import creds as _creds_mod

        # Ensure migration doesn't touch the creds cache (real
        # ~/.claude/.credentials.json could populate __default__).
        monkeypatch.setenv("BYPASS_KEYCHAIN", "1")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        # Stub creds.get_api_key so migration for claude-cli is a no-op
        import open_llm_proxy.creds as _orig_creds

        monkeypatch.setattr(
            _orig_creds,
            "get_api_key",
            lambda account=None: (_ for _ in ()).throw(RuntimeError("no cred")),
        )

        # Populate the default cache with a known stale value
        _creds_mod._cached_key_cache["__default__"] = "stale-default-key"
        _creds_mod._cached_time_cache["__default__"] = 9999999999.0

        # Set up openrouter accounts
        _openrouter_with_stdin(monkeypatch, "sk-or-1\n")
        cli.main(["auth", "add", "openrouter"])
        capsys.readouterr()
        _openrouter_with_stdin(monkeypatch, "sk-or-2\n")
        cli.main(["auth", "add", "openrouter", "--name", "work"])
        capsys.readouterr()

        # Verify stale value is present before `auth use`
        assert _creds_mod._cached_key_cache.get("__default__") == "stale-default-key"

        assert cli.main(["auth", "use", "openrouter", "work"]) == 0

        # After `auth use`, the __default__ cache should be cleared
        assert _creds_mod._cached_key_cache.get("__default__") is None
        assert _creds_mod._cached_time_cache.get("__default__") is None


# ---- auth remove --------------------------------------------------------------


class TestAuthRemove:
    def test_remove_second_then_repoints_active(self, cfg, monkeypatch, capsys):
        _openrouter_with_stdin(monkeypatch, "sk-or-1\n")
        cli.main(["auth", "add", "openrouter"])
        capsys.readouterr()
        _openrouter_with_stdin(monkeypatch, "sk-or-2\n")
        cli.main(["auth", "add", "openrouter", "--name", "work"])
        capsys.readouterr()

        assert cli.main(["auth", "remove", "openrouter", "work"]) == 0
        from open_llm_proxy import account_registry

        accounts = account_registry.list_accounts("openrouter")
        assert len(accounts) == 1
        assert accounts[0].name == "default"
        assert accounts[0].is_active is True

    def test_remove_last_without_force_fails(self, cfg, monkeypatch, capsys):
        _openrouter_with_stdin(monkeypatch, "sk-or-1\n")
        cli.main(["auth", "add", "openrouter"])
        capsys.readouterr()

        assert cli.main(["auth", "remove", "openrouter", "default"]) == 1
        err = capsys.readouterr().err
        assert "Cannot remove last account" in err

    def test_remove_last_with_force_succeeds(self, cfg, monkeypatch, capsys):
        _openrouter_with_stdin(monkeypatch, "sk-or-1\n")
        cli.main(["auth", "add", "openrouter"])
        capsys.readouterr()

        assert cli.main(["auth", "remove", "openrouter", "default", "--force"]) == 0
        from open_llm_proxy import account_registry

        assert account_registry.list_accounts("openrouter") == []

    def test_remove_nonexistent(self, cfg, monkeypatch, capsys):
        _openrouter_with_stdin(monkeypatch, "sk-or-1\n")
        cli.main(["auth", "add", "openrouter"])
        capsys.readouterr()

        assert cli.main(["auth", "remove", "openrouter", "nope"]) == 1


# ---- auth set still works -----------------------------------------------------


class TestAuthSet:
    def test_auth_set_openrouter(self, cfg, monkeypatch, capsys):
        import open_llm_proxy.openrouter_creds

        monkeypatch.setattr(
            open_llm_proxy.openrouter_creds,
            "get_persisted_api_key",
            lambda: (_ for _ in ()).throw(RuntimeError("no cred")),
        )
        saved = []
        monkeypatch.setattr(
            open_llm_proxy.openrouter_creds,
            "save_api_key",
            lambda key: saved.append(key),
        )
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt: "sk-set-key")

        assert cli.main(["auth", "set", "openrouter"]) == 0
        assert saved == ["sk-set-key"]

        # Registry should now have an @default account
        from open_llm_proxy import account_registry

        accounts = account_registry.list_accounts("openrouter")
        assert len(accounts) == 1
        assert accounts[0].name == "default"

    def test_auth_set_openrouter_via_pipe(self, cfg, monkeypatch, capsys):
        import open_llm_proxy.openrouter_creds

        monkeypatch.setattr(
            open_llm_proxy.openrouter_creds,
            "get_persisted_api_key",
            lambda: (_ for _ in ()).throw(RuntimeError("no cred")),
        )
        saved = []
        monkeypatch.setattr(
            open_llm_proxy.openrouter_creds,
            "save_api_key",
            lambda key: saved.append(key),
        )
        fake_stdin = io.StringIO("sk-piped-key\n")
        monkeypatch.setattr(cli.sys, "stdin", fake_stdin)
        monkeypatch.setattr(fake_stdin, "isatty", lambda: False)

        assert cli.main(["auth", "set", "openrouter"]) == 0
        assert saved == ["sk-piped-key"]


# ---- auth check still works ---------------------------------------------------


class TestAuthCheck:
    def test_auth_check_calls_connectivity(self, cfg, monkeypatch, capsys):
        from open_llm_proxy import connectivity

        checked = []
        monkeypatch.setattr(
            connectivity,
            "check_provider",
            lambda p, account=None: (
                checked.append(p),
                (True, "Ready"),
            )[1],
        )

        assert cli.main(["auth", "check"]) == 0
        assert "openrouter" in checked
        assert "opencode" in checked
        assert "github-copilot" in checked
        assert "claude-cli" in checked

    def test_auth_check_specific_provider(self, cfg, monkeypatch, capsys):
        from open_llm_proxy import connectivity

        checked = []
        monkeypatch.setattr(
            connectivity,
            "check_provider",
            lambda p, account=None: (
                checked.append(p),
                (True, "Ready"),
            )[1],
        )

        assert cli.main(["auth", "check", "openrouter"]) == 0
        assert checked == ["openrouter"]


# ---- CRITICAL 1 — named claude-cli OAuth add ---------------------------------


class TestAuthAddClaudeCli:
    def test_named_claude_add_snapshots_credential(self, cfg, monkeypatch, capsys):
        """auth add claude-cli --name work registers a named account whose
        get_api_key(account='work') resolves to the captured token."""
        import json

        from open_llm_proxy import auth_migration

        # Stub migration to be a no-op (re-registering is idempotent)
        monkeypatch.setattr(auth_migration, "migrate_legacy_credentials", lambda: [])
        monkeypatch.setattr(cli, "_run_claude_cli_login", lambda: 0)
        monkeypatch.setattr(cli, "_check_claude_cli", lambda: (True, "credential discoverable"))

        # Create a fake ~/.claude/.credentials.json with OAuth data for capture
        fake_home = cfg.parent
        claude_dir = fake_home / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        creds_data = {
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-named-work",
                "refreshToken": "work-refresh",
                "expiresAt": 9999999999999,
            }
        }
        (claude_dir / ".credentials.json").write_text(json.dumps(creds_data), encoding="utf-8")
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        # First add default account (no --name), then named account
        assert cli.main(["auth", "add", "claude-cli"]) == 0
        capsys.readouterr()
        assert cli.main(["auth", "add", "claude-cli", "--name", "work"]) == 0
        out = capsys.readouterr().out
        assert "work" in out

        from open_llm_proxy import account_registry

        accounts = account_registry.list_accounts("claude-cli")
        assert len(accounts) == 2
        work_acct = next(a for a in accounts if a.name == "work")
        assert work_acct.ref.startswith("accounts/")

        # Verify get_api_key(account="work") resolves the captured token
        from open_llm_proxy import creds as _creds_mod

        _creds_mod._cached_key_cache.clear()
        _creds_mod._cached_time_cache.clear()
        _creds_mod._in_memory_oauth_cache.clear()

        monkeypatch.setenv("BYPASS_KEYCHAIN", "1")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

        token = _creds_mod.get_api_key(account="work")
        assert token == "sk-ant-oat01-named-work"

    def test_named_claude_add_no_credential_errors(self, cfg, monkeypatch, capsys):
        """When the post-login credential can't be captured, the account is
        NOT created and an error is reported."""
        from open_llm_proxy import auth_migration

        monkeypatch.setattr(auth_migration, "migrate_legacy_credentials", lambda: [])

        monkeypatch.setattr(cli, "_run_claude_cli_login", lambda: 0)
        monkeypatch.setattr(cli, "_check_claude_cli", lambda: (True, "credential discoverable"))

        # Stub the credential capture to return None
        monkeypatch.setattr(cli, "_capture_oauth_credential", lambda provider: None)

        # First create the default account
        from open_llm_proxy import account_registry

        assert cli.main(["auth", "add", "claude-cli"]) == 0
        capsys.readouterr()
        assert len(account_registry.list_accounts("claude-cli")) == 1

        # Attempt the named add — should fail because capture returns None
        assert cli.main(["auth", "add", "claude-cli", "--name", "work"]) == 1
        err = capsys.readouterr().err
        assert "Could not capture" in err

        # Only the default account should exist (work was not created)
        accounts = account_registry.list_accounts("claude-cli")
        assert len(accounts) == 1
        assert accounts[0].name == "default"

    def test_default_claude_add_keeps_external_storage(self, cfg, monkeypatch, capsys):
        """Adding the default (first) claude-cli account uses external storage."""
        from open_llm_proxy import auth_migration

        monkeypatch.setattr(auth_migration, "migrate_legacy_credentials", lambda: [])

        monkeypatch.setattr(cli, "_run_claude_cli_login", lambda: 0)
        monkeypatch.setattr(cli, "_check_claude_cli", lambda: (True, "credential discoverable"))

        assert cli.main(["auth", "add", "claude-cli"]) == 0

        from open_llm_proxy import account_registry

        accounts = account_registry.list_accounts("claude-cli")
        assert len(accounts) == 1
        assert accounts[0].name == "default"
        assert accounts[0].storage == "external"
