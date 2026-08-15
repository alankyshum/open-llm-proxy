import pytest

from open_llm_proxy.callbacks import AttachmentContentNormalizationCallback
from open_llm_proxy.content_parts import normalize_content, normalize_messages


def test_pdf_file_data_uri_becomes_placeholder():
    messages = [{"role": "user", "content": [{"type": "file", "filename": "a.pdf", "data": "data:application/pdf;base64,JVBERg=="}]}]
    result = normalize_messages(messages)
    text = result[0]["content"][0]["text"]
    assert "a.pdf" in text and "application/pdf" in text


def test_text_file_data_uri_is_decoded():
    content = [{"type": "file", "filename": "note.txt", "data": "data:text/plain;base64,aGVsbG8="}]
    assert normalize_content(content)[0]["text"].endswith("hello")


@pytest.mark.parametrize("part_type", ["file", "input_image"])
def test_image_data_uri_becomes_image_url(part_type):
    content = [{"type": part_type, "data": "data:image/png;base64,aGVsbG8="}]
    assert normalize_content(content) == [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}]


def test_standard_parts_and_strings_are_identity_noops():
    content = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "https://x"}}]
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
    messages = [{"role": "user", "content": [
        tool_result,
        {"type": "file", "filename": "a.pdf", "data": "data:application/pdf;base64,JVBERg=="},
    ]}]

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
    data = {"messages": [{"role": "user", "content": [{"type": "document", "mime_type": "application/pdf", "filename": "x.pdf"}]}]}
    assert await callback.async_pre_call_hook(None, None, data, "completion") is data
    assert data["messages"][0]["content"][0]["type"] == "text"
    monkeypatch.setenv("OPEN_LLM_PROXY_NORMALIZE_ATTACHMENTS", "false")
    disabled = {"messages": [{"role": "user", "content": [{"type": "file", "data": "data:text/plain,hi"}]}]}
    assert await callback.async_pre_call_hook(None, None, disabled, "completion") is None
