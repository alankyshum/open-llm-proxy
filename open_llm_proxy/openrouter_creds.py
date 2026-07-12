from __future__ import annotations

import os
import shlex
import tempfile
from pathlib import Path


def _env_file() -> Path:
    return Path.home() / ".config" / "open-llm-proxy" / "env"


def get_persisted_api_key() -> str:
    """Read the OpenRouter key available to the background service."""
    env_file = _env_file()
    if env_file.is_file():
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
            if key.strip() != "OPENROUTER_API_KEY":
                continue
            parsed = shlex.split(value.strip())
            if parsed and parsed[0].strip():
                return parsed[0].strip()

    raise RuntimeError("OPENROUTER_API_KEY is absent from ~/.config/open-llm-proxy/env")


def get_api_key() -> str:
    """Reads OPENROUTER_API_KEY from environment or parses ~/.config/open-llm-proxy/env.
    Raises RuntimeError if absent.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if key and key.strip():
        return key.strip()

    return get_persisted_api_key()


def save_api_key(key: str) -> None:
    """Atomically updates/adds OPENROUTER_API_KEY=... to ~/.config/open-llm-proxy/env,
    preserving unrelated lines and setting file permissions to 0600.
    """
    if not key:
        raise ValueError("API key cannot be empty")
    if "\n" in key or "\r" in key:
        raise ValueError("API key cannot contain newline characters")

    env_file = _env_file()
    env_file.parent.mkdir(parents=True, exist_ok=True)

    quoted_key = shlex.quote(key)
    new_line = f"OPENROUTER_API_KEY={quoted_key}"

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
                if temp_line.startswith("OPENROUTER_API_KEY="):
                    is_target = True

            if is_target:
                lines.append(new_line)
                found = True
            else:
                lines.append(line)

    if not found:
        lines.append(new_line)

    new_content = "\n".join(lines) + "\n"

    # Use NamedTemporaryFile in the same directory for atomic replace
    fd, temp_path_str = tempfile.mkstemp(dir=str(env_file.parent), prefix="env_tmp_")
    temp_path = Path(temp_path_str)
    try:
        os.chmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new_content)
        temp_path.replace(env_file)
        # Just to be extremely robust, set 0600 on the final path too
        try:
            os.chmod(str(env_file), 0o600)
        except Exception:
            pass
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise e
