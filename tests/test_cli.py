import subprocess

from open_llm_proxy import cli


def test_status_requires_http_readiness(monkeypatch, capsys):
    launchctl_output = "state = running\n\tpid = 123\n\tstate = active\n"
    monkeypatch.setattr(
        cli,
        "_launchctl",
        lambda *args: subprocess.CompletedProcess(args, 0, launchctl_output, ""),
    )
    monkeypatch.setattr(cli, "_service_is_ready", lambda: False)

    assert cli.main(["status"]) == 1
    output = capsys.readouterr().out
    assert "State:   starting (not ready)" in output
    assert "Health:  unavailable" in output
    assert "state = active" not in output


def test_status_reports_active_only_when_ready(monkeypatch, capsys):
    launchctl_output = "state = running\n\tpid = 123\n"
    monkeypatch.setattr(
        cli,
        "_launchctl",
        lambda *args: subprocess.CompletedProcess(args, 0, launchctl_output, ""),
    )
    monkeypatch.setattr(cli, "_service_is_ready", lambda: True)

    assert cli.main(["status"]) == 0
    output = capsys.readouterr().out
    assert "State:   active" in output
    assert "Health:  ready" in output


def test_help_lists_available_commands(capsys):
    assert cli.main(["help"]) == 0

    output = capsys.readouterr().out
    assert "serve" in output
    assert "setup" in output
    assert "config" in output
    assert "models" in output


def test_help_describes_command_options(capsys):
    assert cli.main(["help", "serve"]) == 0

    output = capsys.readouterr().out
    assert "usage: open-llm-proxy serve" in output
    assert "--host" in output
    assert "--port" in output


def test_serve_delegates_to_server_launcher(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "open_llm_proxy.server_launcher.launch_server",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert cli.main(["serve", "--host", "127.0.0.1", "--port", "9000"]) == 0
    assert len(calls) == 1
    assert calls[0][1]["host"] == "127.0.0.1"
    assert calls[0][1]["port"] == 9000


def test_setup_delegates_to_rate_limit_configuration(monkeypatch, tmp_path):
    calls = []
    config_path = tmp_path / "agent-config.yml"
    monkeypatch.setattr(
        "open_llm_proxy.setup.configure",
        lambda config, interactive, force: calls.append(
            (config, interactive, force)
        ),
    )

    assert (
        cli.main(
            [
                "setup",
                "--config",
                str(config_path),
                "--non-interactive",
                "--force",
            ]
        )
        == 0
    )
    assert calls == [(config_path, False, True)]


def test_config_prints_generated_json(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "agent-config.yml"
    monkeypatch.setattr(
        "open_llm_proxy.config_gen.generate_config",
        lambda path: {"source": path},
    )

    assert (
        cli.main(["config", "--config", str(config_path), "--format", "json"]) == 0
    )
    assert capsys.readouterr().out == f'{{\n  "source": "{config_path}"\n}}\n'


def test_config_reports_generation_error(monkeypatch, capsys):
    def fail(_path):
        raise ValueError("bad config")

    monkeypatch.setattr("open_llm_proxy.config_gen.generate_config", fail)

    assert cli.main(["config"]) == 1
    assert capsys.readouterr().err == "Error: bad config\n"


def test_models_lists_filtered_provider_catalog(monkeypatch, capsys):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "openrouter/moonshotai/kimi-k2.6\n"
                "openrouter/moonshotai/kimi-k2.7-code\n"
                "openrouter/z-ai/glm-5.2\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(cli.subprocess, "run", run)

    assert cli.main(["models", "openrouter", "--search", "K2.7"]) == 0
    assert capsys.readouterr().out == "openrouter/moonshotai/kimi-k2.7-code\n"
    assert calls == [
        (
            ["opencode", "models", "openrouter"],
            {"capture_output": True, "text": True, "check": False},
        )
    ]


def test_models_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="opencode/big-pickle\n", stderr=""
        ),
    )

    assert cli.main(["available-models", "opencode", "--format", "json"]) == 0
    assert capsys.readouterr().out == '[\n  "opencode/big-pickle"\n]\n'


def test_models_reports_missing_opencode(monkeypatch, capsys):
    def missing(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(cli.subprocess, "run", missing)

    assert cli.main(["models"]) == 1
    assert "Install OpenCode" in capsys.readouterr().err


def test_models_reports_catalog_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 2, stdout="", stderr="Unknown provider\n"
        ),
    )

    assert cli.main(["models", "missing-provider"]) == 2
    assert capsys.readouterr().err == (
        "Error: model discovery failed: Unknown provider\n"
    )


def test_auth_all_ok(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_check_openrouter", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_opencode", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_github_copilot", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_claude_cli", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_nvidia", lambda: (True, "credential discoverable"))

    sub_calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: sub_calls.append(args))

    # --no-tui ensures orchestrator path is used even when stdin is a TTY
    assert cli.main(["auth", "--no-tui"]) == 0
    assert not sub_calls
    out = capsys.readouterr().out
    assert "[OK] openrouter: credential discoverable" in out
    assert "[OK] opencode: credential discoverable" in out
    assert "[OK] github-copilot: credential discoverable" in out
    assert "[OK] claude-cli: credential discoverable" in out
    assert "[OK] nvidia: credential discoverable" in out


def test_auth_openrouter_piped(monkeypatch, capsys):
    import io
    or_ok = [False]
    monkeypatch.setattr(cli, "_check_openrouter", lambda: (or_ok[0], "missing" if not or_ok[0] else "credential discoverable"))
    monkeypatch.setattr(cli, "_check_opencode", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_github_copilot", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_claude_cli", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_nvidia", lambda: (True, "credential discoverable"))

    saved_keys = []
    import open_llm_proxy.openrouter_creds
    monkeypatch.setattr(open_llm_proxy.openrouter_creds, "save_api_key", lambda key: (saved_keys.append(key), or_ok.__setitem__(0, True))[0])

    fake_stdin = io.StringIO("piped_secret_key\n")
    monkeypatch.setattr(cli.sys, "stdin", fake_stdin)
    monkeypatch.setattr(fake_stdin, "isatty", lambda: False)

    assert cli.main(["auth", "--no-tui"]) == 0
    assert saved_keys == ["piped_secret_key"]
    out = capsys.readouterr().out
    assert "piped_secret_key" not in out
    assert "[OK] openrouter: credential discoverable" in out


def test_auth_openrouter_tty(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    or_ok = [False]
    monkeypatch.setattr(cli, "_check_openrouter", lambda: (or_ok[0], "missing" if not or_ok[0] else "credential discoverable"))
    monkeypatch.setattr(cli, "_check_opencode", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_github_copilot", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_claude_cli", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_nvidia", lambda: (True, "credential discoverable"))

    saved_keys = []
    import open_llm_proxy.openrouter_creds
    monkeypatch.setattr(
        open_llm_proxy.openrouter_creds,
        "get_api_key",
        lambda: (_ for _ in ()).throw(RuntimeError("missing")),
    )
    monkeypatch.setattr(open_llm_proxy.openrouter_creds, "save_api_key", lambda key: (saved_keys.append(key), or_ok.__setitem__(0, True))[0])
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("getpass.getpass", lambda prompt: "tty_secret_key")

    assert cli.main(["auth", "--no-tui"]) == 0
    assert saved_keys == ["tty_secret_key"]
    out = capsys.readouterr().out
    assert "tty_secret_key" not in out
    assert "[OK] openrouter: credential discoverable" in out


def test_auth_openrouter_empty(monkeypatch, capsys):
    import io
    monkeypatch.setattr(cli, "_check_openrouter", lambda: (False, "missing"))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("   \n"))

    assert cli.main(["auth", "--no-tui"]) == 1
    err = capsys.readouterr().err
    assert "Error: OpenRouter API key cannot be empty" in err


def _stub_all_check_ok(monkeypatch) -> None:
    """Stub all _check_* functions to return found."""
    monkeypatch.setattr(cli, "_check_openrouter", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_opencode", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_github_copilot", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_claude_cli", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_nvidia", lambda: (True, "credential discoverable"))


def test_auth_opencode_login(monkeypatch, capsys):
    _stub_all_check_ok(monkeypatch)

    op_ok = [False]
    monkeypatch.setattr(cli, "_check_opencode", lambda: (op_ok[0], "missing" if not op_ok[0] else "credential discoverable"))

    sub_calls = []
    def mock_run(cmd, **kwargs):
        sub_calls.append(cmd)
        op_ok[0] = True
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", mock_run)

    assert cli.main(["auth", "--no-tui"]) == 0
    assert sub_calls == [["opencode", "auth", "login", "https://opencode.ai"]]
    out = capsys.readouterr().out
    assert "[OK] opencode: credential discoverable" in out


def test_auth_opencode_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_check_openrouter", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_opencode", lambda: (False, "missing"))

    def mock_run(cmd, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(cli.subprocess, "run", mock_run)

    assert cli.main(["auth"]) == 127
    err = capsys.readouterr().err
    assert "Error: opencode command not available" in err


def test_auth_opencode_fails(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_check_openrouter", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_opencode", lambda: (False, "missing"))

    def mock_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(cli.subprocess, "run", mock_run)

    assert cli.main(["auth"]) == 1
    err = capsys.readouterr().err
    assert "Error: opencode auth login failed" in err


def test_auth_opencode_unresolved(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_check_openrouter", lambda: (True, "credential discoverable"))
    monkeypatch.setattr(cli, "_check_opencode", lambda: (False, "missing"))

    def mock_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", mock_run)

    assert cli.main(["auth"]) == 1
    err = capsys.readouterr().err
    assert "Error: OpenCode credential unresolved after authentication" in err


def test_auth_github_copilot_login(monkeypatch, capsys):
    _stub_all_check_ok(monkeypatch)

    cop_ok = [False]
    monkeypatch.setattr(cli, "_check_github_copilot", lambda: (cop_ok[0], "missing" if not cop_ok[0] else "credential discoverable"))

    sub_calls = []
    def mock_run(cmd, **kwargs):
        sub_calls.append(cmd)
        cop_ok[0] = True
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", mock_run)

    assert cli.main(["auth", "--no-tui"]) == 0
    assert sub_calls == [["opencode", "auth", "login", "https://github.com"]]
    out = capsys.readouterr().out
    assert "[OK] github-copilot: credential discoverable" in out


def test_auth_claude_cli_login(monkeypatch, capsys):
    _stub_all_check_ok(monkeypatch)

    cl_ok = [False]
    monkeypatch.setattr(cli, "_check_claude_cli", lambda: (cl_ok[0], "missing" if not cl_ok[0] else "credential discoverable"))

    sub_calls = []
    def mock_run(cmd, **kwargs):
        sub_calls.append(cmd)
        cl_ok[0] = True
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", mock_run)

    assert cli.main(["auth", "--no-tui"]) == 0
    assert sub_calls == [["claude", "auth", "login"]]
    out = capsys.readouterr().out
    assert "[OK] claude-cli: credential discoverable" in out


def test_auth_check_command(monkeypatch, capsys):
    from open_llm_proxy import connectivity
    checked_providers = []
    results = {
        "openrouter": (True, "Ready"),
        "opencode": (False, "Authentication Failed"),
        "github-copilot": (True, "Ready"),
        "claude-cli": (True, "Ready"),
        "nvidia": (True, "Ready"),
    }
    def mock_check(p):
        checked_providers.append(p)
        return results[p]

    monkeypatch.setattr(connectivity, "check_provider", mock_check)

    assert cli.main(["auth", "check"]) == 1
    assert checked_providers == ["openrouter", "opencode", "github-copilot", "claude-cli", "nvidia"]
    captured = capsys.readouterr()
    assert "[FAILED] opencode: Authentication Failed" in captured.err
    assert "[OK] openrouter: Ready" in captured.out


def test_auth_set_command(monkeypatch, capsys):
    saved = []
    import open_llm_proxy.openrouter_creds
    monkeypatch.setattr(open_llm_proxy.openrouter_creds, "save_api_key", lambda key: saved.append(key))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("getpass.getpass", lambda prompt: "manual_set_key")

    assert cli.main(["auth", "set", "openrouter"]) == 0
    assert saved == ["manual_set_key"]
    assert "manual_set_key" not in capsys.readouterr().out


def test_auth_set_opencode(monkeypatch, capsys):
    op_ok = [False]
    monkeypatch.setattr(cli, "_check_opencode", lambda: (op_ok[0], "missing" if not op_ok[0] else "credential discoverable"))

    sub_calls = []
    def mock_run(cmd, **kwargs):
        sub_calls.append(cmd)
        op_ok[0] = True
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", mock_run)

    assert cli.main(["auth", "set", "opencode"]) == 0
    assert sub_calls == [["opencode", "auth", "login", "https://opencode.ai"]]


def test_auth_set_github_copilot(monkeypatch, capsys):
    cop_ok = [False]
    monkeypatch.setattr(cli, "_check_github_copilot", lambda: (cop_ok[0], "missing" if not cop_ok[0] else "credential discoverable"))

    sub_calls = []
    def mock_run(cmd, **kwargs):
        sub_calls.append(cmd)
        cop_ok[0] = True
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", mock_run)

    assert cli.main(["auth", "set", "github-copilot"]) == 0
    assert sub_calls == [["opencode", "auth", "login", "https://github.com"]]


def test_auth_set_claude_cli(monkeypatch, capsys):
    cl_ok = [False]
    monkeypatch.setattr(cli, "_check_claude_cli", lambda: (cl_ok[0], "missing" if not cl_ok[0] else "credential discoverable"))

    sub_calls = []
    def mock_run(cmd, **kwargs):
        sub_calls.append(cmd)
        cl_ok[0] = True
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", mock_run)

    assert cli.main(["auth", "set", "claude-cli"]) == 0
    assert sub_calls == [["claude", "auth", "login"]]
