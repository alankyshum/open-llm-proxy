from open_llm_proxy import cli


def test_help_lists_available_commands(capsys):
    assert cli.main(["help"]) == 0

    output = capsys.readouterr().out
    assert "serve" in output
    assert "setup" in output
    assert "config" in output


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
        lambda host, port: calls.append((host, port)),
    )

    assert cli.main(["serve", "--host", "127.0.0.1", "--port", "9000"]) == 0
    assert calls == [("127.0.0.1", 9000)]


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
