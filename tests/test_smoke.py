def test_import():
    import importlib.metadata

    assert importlib.metadata.version("litellm") is not None


def test_custom_provider_registration():
    import litellm

    assert hasattr(litellm, "custom_provider_map")
    providers = [item.get("provider") for item in litellm.custom_provider_map]
    assert "claude-cli" in providers
    assert "github-copilot" in providers
