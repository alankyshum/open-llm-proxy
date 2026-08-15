"""Spool non-renderable attachment bytes to disk and hand back a path.

Upstream chat providers accept only ``text`` and ``image_url`` content parts, so
an attachment such as a PDF cannot be forwarded inline. Rather than teaching the
proxy to parse arbitrary formats, we persist the raw bytes and let the agent use
its own file-reading tools on the resulting absolute path.

Everything here is best effort: any failure returns ``None`` so the caller can
fall back to a purely descriptive placeholder.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import re
import tempfile
import time
from pathlib import Path

log = logging.getLogger("open_llm_proxy.attachment_spool")

_DEFAULT_DIR = "~/.local/share/open-llm-proxy/attachments"
_DEFAULT_RETENTION_DAYS = 7.0
_DIR_MODE = 0o700
_FILE_MODE = 0o600
_HASH_PREFIX_LEN = 16
_MAX_NAME_LEN = 96

# Anything outside this set collapses to "_" so the basename is always safe.
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def spooling_enabled() -> bool:
    """Kill switch: ``OPEN_LLM_PROXY_SPOOL_ATTACHMENTS=0`` disables spooling."""
    return _env_flag("OPEN_LLM_PROXY_SPOOL_ATTACHMENTS", True)


def spool_dir() -> Path:
    raw = os.environ.get("OPEN_LLM_PROXY_ATTACHMENT_DIR") or _DEFAULT_DIR
    return Path(raw).expanduser()


def _retention_seconds() -> float | None:
    raw = os.environ.get("OPEN_LLM_PROXY_ATTACHMENT_RETENTION_DAYS")
    days = _DEFAULT_RETENTION_DAYS
    if raw is not None:
        try:
            days = float(raw)
        except (TypeError, ValueError):
            days = _DEFAULT_RETENTION_DAYS
    if days <= 0:
        return None  # Retention disabled; never prune.
    return days * 86400.0


def _extension_for(mime: str | None) -> str:
    if mime:
        guessed = mimetypes.guess_extension(mime.split(";")[0].strip().lower())
        if guessed:
            return guessed
    return ".bin"


def _sanitize(filename: str | None, mime: str | None) -> str:
    """Reduce an arbitrary client-supplied name to a safe, non-empty basename."""
    candidate = ""
    if filename:
        # Defeat traversal and absolute paths from either path flavour.
        candidate = str(filename).replace("\\", "/").split("/")[-1].strip()
    candidate = _UNSAFE_CHARS.sub("_", candidate).strip("._-")
    if not candidate:
        candidate = f"attachment{_extension_for(mime)}"
    if len(candidate) > _MAX_NAME_LEN:
        stem, dot, ext = candidate.rpartition(".")
        if dot and len(ext) <= 10:
            keep = _MAX_NAME_LEN - len(ext) - 1
            candidate = f"{stem[:keep]}.{ext}"
        else:
            candidate = candidate[:_MAX_NAME_LEN]
    return candidate


def spooled_name(data: bytes, filename: str | None, mime: str | None) -> str:
    """Content-addressed basename: ``<sha256[:16]>-<sanitized-original-name>``."""
    digest = hashlib.sha256(data).hexdigest()[:_HASH_PREFIX_LEN]
    return f"{digest}-{_sanitize(filename, mime)}"


def prune_spool_dir(directory: Path, retention_seconds: float | None = None) -> int:
    """Delete spooled files older than the retention window. Never raises."""
    removed = 0
    try:
        if retention_seconds is None:
            retention_seconds = _retention_seconds()
        if retention_seconds is None:
            return 0
        cutoff = time.time() - retention_seconds
        for entry in directory.iterdir():
            try:
                if not entry.is_file():
                    continue
                if entry.stat().st_mtime < cutoff:
                    entry.unlink()
                    removed += 1
            except OSError:
                continue
    except Exception:
        log.debug("attachment spool pruning skipped", exc_info=True)
    return removed


def spool_attachment(
    data: bytes, filename: str | None = None, mime: str | None = None
) -> Path | None:
    """Persist ``data`` and return its absolute path, or ``None`` on any failure.

    The name is deterministic and content addressed, so a fallback chain that
    retries the same request against the next model reuses the existing file
    instead of writing a duplicate.
    """
    if not spooling_enabled():
        return None
    if not isinstance(data, (bytes, bytearray)) or not data:
        return None
    data = bytes(data)
    try:
        directory = spool_dir()
        directory.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        try:
            os.chmod(directory, _DIR_MODE)
        except OSError:
            pass

        target = (directory / spooled_name(data, filename, mime)).resolve()

        # Same content hash and same size means the payload is already spooled.
        try:
            if target.is_file() and target.stat().st_size == len(data):
                prune_spool_dir(directory)
                return target
        except OSError:
            pass

        handle, temp_name = tempfile.mkstemp(dir=str(directory), prefix=".tmp-")
        temp_path = Path(temp_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temp_path, _FILE_MODE)
            os.replace(temp_path, target)
        except Exception:
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise

        prune_spool_dir(directory)
        return target
    except Exception:
        log.debug("attachment spooling failed", exc_info=True)
        return None
