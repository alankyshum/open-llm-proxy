"""Suite-wide isolation from developer configuration and credentials."""

from __future__ import annotations

import ast
import os
import re
import tempfile
from pathlib import Path

import pytest

_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]+$")


def _source_environment_names() -> set[str]:
    names: set[str] = set()
    source_root = Path(__file__).parents[1] / "open_llm_proxy"
    for path in source_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if _ENV_NAME.fullmatch(node.value):
                    names.add(node.value)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "getenv" and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        names.add(arg.value)
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
                if node.value.attr == "environ":
                    key = node.slice
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        names.add(key.value)
    return names


SOURCE_ENVIRONMENT_NAMES = frozenset(_source_environment_names())


def _clear_source_environment() -> None:
    for name in SOURCE_ENVIRONMENT_NAMES:
        os.environ.pop(name, None)


def _isolate_environment(config_dir: Path) -> None:
    _clear_source_environment()
    os.environ["XDG_CONFIG_HOME"] = str(config_dir / "xdg")
    os.environ["BYPASS_KEYCHAIN"] = "1"
    os.environ["HOME"] = str(config_dir / "home")


# Import-time probes in live-test modules must not see the developer's credentials.
_IMPORT_CONFIG_DIR = Path(tempfile.mkdtemp(prefix="open-llm-proxy-tests-"))
_isolate_environment(_IMPORT_CONFIG_DIR)


@pytest.fixture(scope="session", autouse=True)
def isolate_import_environment():
    """Give the suite an isolated home/config root and no ambient secrets."""
    _isolate_environment(_IMPORT_CONFIG_DIR)
    yield


@pytest.fixture(autouse=True)
def isolate_test_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Give every test an isolated home and no ambient source environment."""
    for name in SOURCE_ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("OLP_CONFIG_DIR", raising=False)
    monkeypatch.delenv("OPEN_LLM_PROXY_CONFIG_DIR", raising=False)
    monkeypatch.delenv("OPEN_LLM_PROXY_CONFIG", raising=False)
    monkeypatch.delenv("BYPASS_KEYCHAIN", raising=False)
    yield
