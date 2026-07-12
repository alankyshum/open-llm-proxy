from __future__ import annotations

import os
import shlex
import tempfile
from pathlib import Path

from open_llm_proxy import account_registry


def _env_file() -> Path:
    return account_registry.CONFIG_DIR() / "env"


def get_env_key(name: str) -> str | None:
    """Resolve *name*: env var first, then ~/.config/open-llm-proxy/env.

    Returns the value or ``None`` if not found.
    """
    env_val = os.environ.get(name)
    if env_val and env_val.strip():
        return env_val.strip()

    env_file = _env_file()
    if not env_file.is_file():
        return None
    content = env_file.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        parsed = shlex.split(value.strip())
        if parsed and parsed[0].strip():
            return parsed[0].strip()
    return None


def set_env_key(name: str, value: str) -> None:
    """Atomically set *name=value* in ~/.config/open-llm-proxy/env (0600).

    Preserves unrelated lines.  Raises ``ValueError`` if *value* is empty
    or contains newlines.
    """
    if not value:
        raise ValueError(f"{name} value cannot be empty")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{name} value cannot contain newline characters")

    env_file = _env_file()
    env_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(str(env_file.parent), 0o700)
    except OSError as e:
        raise RuntimeError(
            f"Failed to set permissions on config directory: {e}"
        ) from e

    quoted_value = shlex.quote(value)
    new_line = f"{name}={quoted_value}"

    lines: list[str] = []
    found = False

    if env_file.is_file():
        content = env_file.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            is_target = False
            if not stripped.startswith("#"):
                temp_line = stripped
                if temp_line.startswith("export "):
                    temp_line = temp_line[len("export "):].strip()
                if temp_line.startswith(f"{name}="):
                    is_target = True

            if is_target:
                lines.append(new_line)
                found = True
            else:
                lines.append(line)

    if not found:
        lines.append(new_line)

    new_content = "\n".join(lines) + "\n"

    fd, temp_path_str = tempfile.mkstemp(dir=str(env_file.parent), prefix="env_tmp_")
    temp_path = Path(temp_path_str)
    try:
        os.chmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new_content)
        temp_path.replace(env_file)
        try:
            os.chmod(str(env_file), 0o600)
        except OSError as e:
            raise RuntimeError(
                f"Failed to set permissions on env file: {e}"
            ) from e
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise e
