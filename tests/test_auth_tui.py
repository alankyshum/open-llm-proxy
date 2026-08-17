from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---- Fixtures -----------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set OLP_CONFIG_DIR to a temp directory."""
    d = tmp_path / "olp_config"
    d.mkdir()
    monkeypatch.setenv("OLP_CONFIG_DIR", str(d))
    return d


# ---- Questionary Mock ---------------------------------------------------------


def _install_questionary(monkeypatch, responses: list):
    """Build a fake ``questionary`` module and insert it into sys.modules.

    auth_tui.run_auth_tui() does ``import questionary`` inside the function
    body, so we monkey-patch ``sys.modules['questionary']`` before the call.

    ``responses`` is a list of return values consumed in order by
    ``select(...).ask()``, ``text(...).ask()``, ``password(...).ask()``,
    and ``confirm(...).ask()``.
    """
    import sys

    class _Select:
        def __init__(self, **kw):
            pass

        @staticmethod
        def ask():
            return responses.pop(0) if responses else None

    class _Text:
        def __init__(self, **kw):
            self._validate = kw.get("validate")

        @staticmethod
        def ask():
            return responses.pop(0) if responses else None

    class _Password:
        def __init__(self, **kw):
            pass

        @staticmethod
        def ask():
            return responses.pop(0) if responses else None

    class _Confirm:
        def __init__(self, **kw):
            pass

        @staticmethod
        def ask():
            return responses.pop(0) if responses else None

    q = MagicMock()
    q.select = lambda text, choices=None, **kw: _Select()
    q.text = lambda text, validate=None, **kw: _Text(validate=validate)
    q.password = lambda text, **kw: _Password()
    q.confirm = lambda text, default=False, **kw: _Confirm()
    q.Choice = lambda title, value: value

    monkeypatch.setitem(sys.modules, "questionary", q)
    return q


def _make_stub_migration_skipped(monkeypatch):
    """Prevent migration from finding any legacy credentials."""
    import open_llm_proxy.copilot_creds
    import open_llm_proxy.creds
    import open_llm_proxy.nvidia_creds
    import open_llm_proxy.opencode_creds
    import open_llm_proxy.openrouter_creds

    def _raise(*args):
        raise RuntimeError("no cred")

    monkeypatch.setattr(open_llm_proxy.openrouter_creds, "get_persisted_api_key", _raise)
    monkeypatch.setattr(open_llm_proxy.opencode_creds, "get_opencode_api_key", _raise)
    monkeypatch.setattr(open_llm_proxy.copilot_creds, "get_oauth_token", _raise)
    monkeypatch.setattr(open_llm_proxy.creds, "get_api_key", _raise)
    monkeypatch.setattr(open_llm_proxy.nvidia_creds, "get_api_key", _raise)


def _stub_connectivity_ok(monkeypatch):
    """Make all connectivity checks return Ready."""
    from open_llm_proxy import connectivity

    monkeypatch.setattr(
        connectivity,
        "check_provider",
        lambda p, account=None: (True, "Ready"),
    )


# ---- Tests --------------------------------------------------------------------


class TestTUIDispatch:
    def test_non_tty_falls_back(self, cfg, monkeypatch):
        """Non-TTY: run_auth_tui returns 127 sentinel, bare auth calls orchestrator."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        from open_llm_proxy.auth_tui import run_auth_tui

        assert run_auth_tui() == 127

    def test_bare_auth_no_tui_flag_calls_orchestrator(self, cfg, monkeypatch, capsys):
        """``auth --no-tui`` does not invoke the TUI."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        from open_llm_proxy import cli, connectivity

        checked = []
        monkeypatch.setattr(
            connectivity,
            "check_provider",
            lambda p, account=None: (checked.append(p), (True, "Ready"))[1],
        )
        # Stub orchestrator's check functions so they all report found
        monkeypatch.setattr(cli, "_check_openrouter", lambda: (True, "credential discoverable"))
        monkeypatch.setattr(cli, "_check_opencode", lambda: (True, "credential discoverable"))
        monkeypatch.setattr(cli, "_check_github_copilot", lambda: (True, "credential discoverable"))
        monkeypatch.setattr(cli, "_check_claude_cli", lambda: (True, "credential discoverable"))
        monkeypatch.setattr(cli, "_check_nvidia", lambda: (True, "credential discoverable"))

        # isatty should be True but --no-tui bypasses the TUI
        assert cli.main(["auth", "--no-tui"]) == 0
        out = capsys.readouterr().out
        assert "[OK] openrouter: credential discoverable" in out
        assert "[OK] nvidia: credential discoverable" in out

    def test_bare_auth_subcommand_still_routes(self, cfg, monkeypatch, capsys):
        """``auth accounts`` still routes to the accounts handler, not TUI."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        import open_llm_proxy.copilot_creds
        import open_llm_proxy.creds
        import open_llm_proxy.nvidia_creds
        import open_llm_proxy.opencode_creds
        import open_llm_proxy.openrouter_creds
        from open_llm_proxy import cli

        def _raise(*args):
            raise RuntimeError("no cred")

        monkeypatch.setattr(open_llm_proxy.openrouter_creds, "get_persisted_api_key", _raise)
        monkeypatch.setattr(open_llm_proxy.opencode_creds, "get_opencode_api_key", _raise)
        monkeypatch.setattr(open_llm_proxy.copilot_creds, "get_oauth_token", _raise)
        monkeypatch.setattr(open_llm_proxy.creds, "get_api_key", _raise)
        monkeypatch.setattr(open_llm_proxy.nvidia_creds, "get_api_key", _raise)

        assert cli.main(["auth", "accounts"]) == 0
        out = capsys.readouterr().out
        assert "No credentials configured" in out


class TestTUIAdd:
    def test_add_nvidia_api_key_default(self, cfg, monkeypatch):
        """Add nvidia api-key path registers @default."""
        _make_stub_migration_skipped(monkeypatch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        responses = [
            "add",  # Top-level action
            "nvidia",  # Provider select
            "nv-key-123",  # Password input
            None,  # Next action -> Ctrl-C -> quit
        ]
        _install_questionary(monkeypatch, responses)

        from open_llm_proxy.auth_tui import run_auth_tui

        rc = run_auth_tui()
        assert rc == 0

    def test_add_nvidia_second_account_requires_name(self, cfg, monkeypatch):
        """Adding a second nvidia account prompts for a name."""
        _make_stub_migration_skipped(monkeypatch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        from open_llm_proxy import account_registry

        # Pre-seed an existing account
        account_registry.add_account("nvidia", "default", storage="env-line", ref="NVIDIA_API_KEY")

        # Mock env_creds.set_env_key to prevent real file writes
        import open_llm_proxy.env_creds

        monkeypatch.setattr(open_llm_proxy.env_creds, "set_env_key", lambda name, value: None)

        responses = [
            "add",  # Top-level action
            "nvidia",  # Provider select
            "work",  # Name (text prompt)
            "nv-key-work",  # Password
            None,  # Next action -> Ctrl-C -> quit
        ]
        _install_questionary(monkeypatch, responses)

        from open_llm_proxy.auth_tui import run_auth_tui

        rc = run_auth_tui()
        assert rc == 0

        accounts = account_registry.list_accounts("nvidia")
        assert len(accounts) == 2
        names = {a.name for a in accounts}
        assert names == {"default", "work"}

    def test_add_opencode_oauth(self, cfg, monkeypatch):
        """Adding opencode (oauth-cli) confirms then runs login helper."""
        _make_stub_migration_skipped(monkeypatch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        from open_llm_proxy import cli as _cli

        login_called = []
        monkeypatch.setattr(_cli, "_run_opencode_login", lambda: (login_called.append(1), 0)[1])
        monkeypatch.setattr(_cli, "_check_opencode", lambda: (True, "credential discoverable"))

        responses = [
            "add",
            "opencode",
            True,  # Confirm: yes, run login helper
            None,  # Next action -> Ctrl-C -> quit
        ]
        _install_questionary(monkeypatch, responses)

        from open_llm_proxy.auth_tui import run_auth_tui

        rc = run_auth_tui()
        assert rc == 0
        assert login_called == [1]


class TestTUIList:
    def test_list_accounts_no_creds(self, cfg, monkeypatch, capsys):
        _make_stub_migration_skipped(monkeypatch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        responses = [
            "list",
            None,  # Ctrl-C to quit after listing
        ]
        _install_questionary(monkeypatch, responses)

        from open_llm_proxy.auth_tui import run_auth_tui

        rc = run_auth_tui()
        assert rc == 0
        out = capsys.readouterr().out
        assert "No credentials configured" in out


class TestTUISwitch:
    def test_switch_no_accounts(self, cfg, monkeypatch, capsys):
        _make_stub_migration_skipped(monkeypatch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        responses = [
            "switch",
            None,  # Ctrl-C to quit
        ]
        _install_questionary(monkeypatch, responses)

        from open_llm_proxy.auth_tui import run_auth_tui

        rc = run_auth_tui()
        assert rc == 1
        assert "No accounts to switch" in capsys.readouterr().err


class TestTUIRemove:
    def test_remove_last_with_confirm(self, cfg, monkeypatch, capsys):
        _make_stub_migration_skipped(monkeypatch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        from open_llm_proxy import account_registry

        account_registry.add_account("nvidia", "default", storage="env-line", ref="NVIDIA_API_KEY")

        responses = [
            "remove",
            "nvidia",
            "default",  # select the only account
            True,  # confirm removal
            None,  # Ctrl-C to quit
        ]
        _install_questionary(monkeypatch, responses)

        from open_llm_proxy.auth_tui import run_auth_tui

        rc = run_auth_tui()
        assert rc == 0
        assert account_registry.list_accounts("nvidia") == []


class TestTUIRename:
    def test_rename_no_multi_account(self, cfg, monkeypatch, capsys):
        _make_stub_migration_skipped(monkeypatch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        from open_llm_proxy import account_registry

        account_registry.add_account("nvidia", "default", storage="env-line", ref="NVIDIA_API_KEY")
        # Only 1 account, rename should be blocked
        responses = [
            "rename",
            None,  # Ctrl-C to quit
        ]
        _install_questionary(monkeypatch, responses)

        from open_llm_proxy.auth_tui import run_auth_tui

        rc = run_auth_tui()
        assert rc == 1
        err = capsys.readouterr().err
        assert "multiple accounts" in err


class TestTUIIntegration:
    def test_quit_from_menu(self, cfg, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        _make_stub_migration_skipped(monkeypatch)

        responses = ["quit"]
        _install_questionary(monkeypatch, responses)

        from open_llm_proxy.auth_tui import run_auth_tui

        assert run_auth_tui() == 0

    def test_ctrl_c_at_menu_returns_zero(self, cfg, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        _make_stub_migration_skipped(monkeypatch)

        responses: list = [None]  # None simulates Ctrl-C
        _install_questionary(monkeypatch, responses)

        from open_llm_proxy.auth_tui import run_auth_tui

        assert run_auth_tui() == 0
