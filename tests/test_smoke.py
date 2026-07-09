def test_import():
    import litellm
    assert litellm.__version__ is not None
