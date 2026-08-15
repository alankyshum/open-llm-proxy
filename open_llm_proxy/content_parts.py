"""Model-agnostic normalization for OpenAI chat message content parts."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any
from urllib.parse import unquote_to_bytes


_TEXT_MIME_TYPES = {"application/json", "application/xml"}
# Tool parts are preserved because the downstream translator handles their pairing.
_TRANSLATOR_PASSTHROUGH_PART_TYPES = {"toolresult", "tooluse"}


def _walk(value: Any, depth: int = 0):
    """Yield dict keys and values, without trusting provider-specific shapes."""
    if not isinstance(value, dict) or depth > 4:
        return
    for key, child in value.items():
        yield key, child
        if isinstance(child, dict):
            yield from _walk(child, depth + 1)


def _normal_key(key: Any) -> str:
    return str(key).replace("_", "").replace("-", "").lower()


def _first_string(part: dict, names: set[str]) -> str | None:
    for key, value in _walk(part):
        if _normal_key(key) in names and isinstance(value, str) and value:
            return value
    return None


def _data_uri(value: Any) -> tuple[str | None, bytes | None]:
    """Return mime and decoded payload for a data URI, if it is valid."""
    if not isinstance(value, str) or not value.lower().startswith("data:"):
        return None, None
    try:
        header, payload = value.split(",", 1)
        metadata = header[5:]
        fields = metadata.split(";")
        mime = fields[0].lower() or None
        if "base64" in {field.lower() for field in fields[1:]}:
            return mime, base64.b64decode(payload, validate=True)
        return mime, unquote_to_bytes(payload)
    except (ValueError, binascii.Error, UnicodeError):
        return None, None


def _is_text_mime(mime: str | None) -> bool:
    mime = (mime or "").lower()
    return (
        mime.startswith("text/")
        or mime in _TEXT_MIME_TYPES
        or mime.endswith("+json")
        or mime.endswith("+xml")
    )


def _raw_base64(value: str) -> bytes | None:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None


def _attachment_header(filename: str | None, mime: str | None) -> str:
    label = filename or mime
    return f"[attachment: {label}]\n" if label else ""


def _normalize_part(part: Any) -> dict:
    """Convert one arbitrary part to one of the two OpenAI content part types."""
    try:
        if not isinstance(part, dict):
            return {"type": "text", "text": str(part)}

        part_type = part.get("type")
        if _normal_key(part_type) in _TRANSLATOR_PASSTHROUGH_PART_TYPES:
            return part
        if part_type == "text":
            if "text" in part:
                return part
            copied = dict(part)
            copied["text"] = ""
            return copied
        if part_type == "image_url":
            return part

        mime = _first_string(part, {"mimetype", "mediatype", "contenttype"})
        filename = _first_string(part, {"filename", "name", "fileid"})
        text = _first_string(part, {"text"})

        # Prefer explicit URL/data fields, including image_url: {url: ...}.
        value = _first_string(
            part,
            {"url", "uri", "data", "filedata", "imageurl", "base64"},
        )
        uri_mime, uri_bytes = _data_uri(value)
        mime = (uri_mime or mime or "").lower() or None

        if mime and mime.startswith("image/"):
            image_value = value
            if image_value and not image_value.lower().startswith("data:"):
                raw = _raw_base64(image_value)
                if raw is not None:
                    image_value = f"data:{mime};base64,{image_value}"
            if image_value:
                return {"type": "image_url", "image_url": {"url": image_value}}

        if text is None and _is_text_mime(mime) and uri_bytes is not None:
            try:
                text = uri_bytes.decode("utf-8")
            except UnicodeDecodeError:
                pass
        if text is not None:
            return {
                "type": "text",
                "text": _attachment_header(filename, mime) + text,
            }

        byte_count = len(uri_bytes) if uri_bytes is not None else None
        if byte_count is None and isinstance(value, str) and value:
            decoded = _raw_base64(value)
            byte_count = len(decoded) if decoded is not None else len(value.encode("utf-8"))
        label = filename or "attachment"
        mime_label = mime or "unknown mime"
        size = f", {byte_count} bytes" if byte_count is not None else ""
        return {
            "type": "text",
            "text": (
                f"[attachment: {label} ({mime_label}){size} — "
                "content not inline-renderable]"
            ),
        }
    except Exception:
        # An upstream client must never be able to make the request hook fail.
        try:
            rendered = json.dumps(part, default=str)
        except Exception:
            rendered = str(part)
        return {"type": "text", "text": rendered}


def normalize_content(content: Any) -> Any:
    """Normalize a list of content parts, retaining object identity for no-ops."""
    if isinstance(content, str) or not isinstance(content, list):
        return content
    normalized: list[dict] = []
    changed = False
    for part in content:
        replacement = _normalize_part(part)
        normalized.append(replacement)
        if replacement is not part:
            changed = True
    return normalized if changed else content


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Return copied messages only when their content needs normalization."""
    if not isinstance(messages, list):
        return messages
    normalized: list[dict] = []
    changed = False
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue
        content = normalize_content(message.get("content"))
        if content is message.get("content"):
            normalized.append(message)
            continue
        copied = dict(message)
        copied["content"] = content
        normalized.append(copied)
        changed = True
    return normalized if changed else messages
