from open_llm_proxy.setup import _choose_plan, configure


def test_choose_plan_rejects_out_of_range_input(monkeypatch):
    answers = iter(["0", "2"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert _choose_plan("claude-cli", "free") == "pro"


def test_non_interactive_setup_uses_configured_plans(tmp_path):
    config_path = tmp_path / "agent-config.yml"
    database_path = tmp_path / "state.sqlite3"
    config_path.write_text(
        f"""
rate_limit_policy:
  database: {database_path}
  plans:
    claude-cli: pro
    google: free
opencode:
  settings:
    model: "open-llm-proxy/[claude-cli/claude-sonnet-5,google/gemini-3.5-flash]"
"""
    )

    store = configure(config_path, interactive=False)

    assert store.configured_plan("claude-cli")[0] == "pro"
    assert store.configured_plan("google")[0] == "free"
    assert {(row["provider"], row["model"]) for row in store.inventory()} == {
        ("claude-cli", "claude-sonnet-5"),
        ("google", "gemini-3.5-flash"),
    }
