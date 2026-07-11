import subprocess

from open_llm_proxy import cli


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
