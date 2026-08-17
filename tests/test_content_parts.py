import pytest

from open_llm_proxy.callbacks import AttachmentContentNormalizationCallback
from open_llm_proxy.content_parts import normalize_content, normalize_messages


@pytest.fixture(autouse=True)
def isolated_spool_dir(tmp_path, monkeypatch):
    """Keep every test in this module off the real spool directory."""
    target = tmp_path / "attachments"
    monkeypatch.setenv("OPEN_LLM_PROXY_ATTACHMENT_DIR", str(target))
    monkeypatch.delenv("OPEN_LLM_PROXY_SPOOL_ATTACHMENTS", raising=False)
    monkeypatch.delenv("OPEN_LLM_PROXY_ATTACHMENT_RETENTION_DAYS", raising=False)
    return target


def test_pdf_file_data_uri_becomes_placeholder():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "filename": "a.pdf",
                    "data": "data:application/pdf;base64,JVBERg==",
                }
            ],
        }
    ]
    result = normalize_messages(messages)
    text = result[0]["content"][0]["text"]
    assert "a.pdf" in text and "application/pdf" in text


def test_text_file_data_uri_is_decoded():
    content = [{"type": "file", "filename": "note.txt", "data": "data:text/plain;base64,aGVsbG8="}]
    assert normalize_content(content)[0]["text"].endswith("hello")


@pytest.mark.parametrize("part_type", ["file", "input_image"])
def test_image_data_uri_becomes_image_url(part_type):
    content = [{"type": part_type, "data": "data:image/png;base64,aGVsbG8="}]
    assert normalize_content(content) == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
    ]


def test_standard_parts_and_strings_are_identity_noops():
    content = [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "https://x"}},
    ]
    messages = [{"role": "user", "content": content, "tool_calls": []}]
    assert normalize_content(content) is content
    assert normalize_messages(messages) is messages
    assert normalize_content("hello") == "hello"


@pytest.mark.parametrize("part_type", ["tool_result", "tool-result"])
def test_tool_result_parts_are_identity_noops(part_type):
    part = {"type": part_type, "tool_use_id": "tool-1", "content": "result"}
    messages = [{"role": "user", "content": [part]}]

    result = normalize_messages(messages)

    assert result is messages
    assert result[0]["content"][0] is part


def test_tool_result_is_preserved_while_file_is_normalized():
    tool_result = {"type": "tool_result", "tool_call_id": "tool-1", "content": "result"}
    messages = [
        {
            "role": "user",
            "content": [
                tool_result,
                {
                    "type": "file",
                    "filename": "a.pdf",
                    "data": "data:application/pdf;base64,JVBERg==",
                },
            ],
        }
    ]

    result = normalize_messages(messages)

    assert result is not messages
    assert result[0]["content"][0] is tool_result
    assert result[0]["content"][1]["type"] == "text"


def test_malformed_part_does_not_raise_and_message_keys_are_preserved():
    messages = [{"role": "tool", "name": "n", "tool_call_id": "id", "content": [object()]}]
    result = normalize_messages(messages)
    assert result[0]["name"] == "n" and result[0]["tool_call_id"] == "id"
    assert result[0]["content"][0]["type"] == "text"


@pytest.mark.anyio
async def test_attachment_callback_rewrites_and_honors_disable(monkeypatch):
    callback = AttachmentContentNormalizationCallback()
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "document", "mime_type": "application/pdf", "filename": "x.pdf"}
                ],
            }
        ]
    }
    assert await callback.async_pre_call_hook(None, None, data, "completion") is data
    assert data["messages"][0]["content"][0]["type"] == "text"
    monkeypatch.setenv("OPEN_LLM_PROXY_NORMALIZE_ATTACHMENTS", "false")
    disabled = {
        "messages": [{"role": "user", "content": [{"type": "file", "data": "data:text/plain,hi"}]}]
    }
    assert await callback.async_pre_call_hook(None, None, disabled, "completion") is None


def _pdf_part(filename="invoice.pdf", body=b"%PDF-1.4 spooled body"):
    import base64

    encoded = base64.b64encode(body).decode()
    return {
        "type": "file",
        "file": {
            "filename": filename,
            "mime_type": "application/pdf",
            "file_data": f"data:application/pdf;base64,{encoded}",
        },
    }


def test_pdf_is_spooled_to_disk_and_path_appears_in_text(isolated_spool_dir):
    body = b"%PDF-1.4 spooled body"

    text = normalize_content([_pdf_part()])[0]["text"]

    spooled = list(isolated_spool_dir.iterdir())
    assert len(spooled) == 1
    assert spooled[0].read_bytes() == body
    assert str(spooled[0]) in text
    assert "invoice.pdf" in text and "application/pdf" in text
    assert f"{len(body)} bytes" in text
    assert "Read this file from disk" in text
    assert "not inline-renderable" not in text


def test_raw_base64_field_without_data_uri_is_also_spooled(isolated_spool_dir):
    import base64

    body = b"\x89binary-not-an-image"
    part = {
        "type": "file",
        "filename": "blob.dat",
        "mime_type": "application/octet-stream",
        "data": base64.b64encode(body).decode(),
    }

    text = normalize_content([part])[0]["text"]

    spooled = list(isolated_spool_dir.iterdir())
    assert len(spooled) == 1 and spooled[0].read_bytes() == body
    assert str(spooled[0]) in text


def test_retrying_the_same_attachment_reuses_one_spooled_path(isolated_spool_dir):
    first = normalize_content([_pdf_part()])[0]["text"]
    second = normalize_content([_pdf_part()])[0]["text"]

    assert first == second
    assert len(list(isolated_spool_dir.iterdir())) == 1


def test_spool_failure_falls_back_to_the_legacy_placeholder(monkeypatch):
    monkeypatch.setattr("open_llm_proxy.content_parts.spool_attachment", lambda *a, **k: None)

    text = normalize_content([_pdf_part()])[0]["text"]

    assert "content not inline-renderable" in text
    assert "Saved to:" not in text


def test_kill_switch_restores_the_legacy_placeholder(monkeypatch, isolated_spool_dir):
    monkeypatch.setenv("OPEN_LLM_PROXY_SPOOL_ATTACHMENTS", "0")

    text = normalize_content([_pdf_part()])[0]["text"]

    assert "content not inline-renderable" in text
    assert not isolated_spool_dir.exists() or not list(isolated_spool_dir.iterdir())


def test_text_and_image_attachments_are_never_spooled(isolated_spool_dir):
    content = [
        {"type": "file", "filename": "note.txt", "data": "data:text/plain;base64,aGVsbG8="},
        {"type": "file", "data": "data:image/png;base64,aGVsbG8="},
    ]

    normalized = normalize_content(content)

    assert normalized[0]["text"].endswith("hello")
    assert normalized[1]["type"] == "image_url"
    assert not isolated_spool_dir.exists() or not list(isolated_spool_dir.iterdir())
