import os
from pathlib import Path

import pytest

from open_llm_proxy import attachment_spool
from open_llm_proxy.attachment_spool import (
    prune_spool_dir,
    spool_attachment,
    spool_dir,
    spooled_name,
)


@pytest.fixture(autouse=True)
def isolated_spool_dir(tmp_path, monkeypatch):
    target = tmp_path / "attachments"
    monkeypatch.setenv("OPEN_LLM_PROXY_ATTACHMENT_DIR", str(target))
    monkeypatch.delenv("OPEN_LLM_PROXY_SPOOL_ATTACHMENTS", raising=False)
    monkeypatch.delenv("OPEN_LLM_PROXY_ATTACHMENT_RETENTION_DAYS", raising=False)
    return target


def test_spooled_file_holds_exact_bytes_with_content_addressed_name():
    data = b"%PDF-1.4 fake pdf body"

    path = spool_attachment(data, "invoice.pdf", "application/pdf")

    assert path is not None
    assert path.is_file()
    assert path.read_bytes() == data
    assert path.name == spooled_name(data, "invoice.pdf", "application/pdf")
    assert path.name.endswith("-invoice.pdf")
    assert path.is_absolute()


def test_directory_and_file_permissions_are_private(isolated_spool_dir):
    path = spool_attachment(b"secret", "s.pdf", "application/pdf")

    assert path is not None
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert oct(isolated_spool_dir.stat().st_mode & 0o777) == "0o700"


def test_same_attachment_twice_is_deterministic_and_does_not_duplicate(
    isolated_spool_dir,
):
    data = b"%PDF-1.4 retry me"

    first = spool_attachment(data, "a.pdf", "application/pdf")
    second = spool_attachment(data, "a.pdf", "application/pdf")

    assert first == second
    assert list(isolated_spool_dir.iterdir()) == [first]


def test_different_bytes_get_different_paths():
    one = spool_attachment(b"aaaa", "a.pdf", "application/pdf")
    two = spool_attachment(b"bbbb", "a.pdf", "application/pdf")

    assert one != two
    assert one.is_file() and two.is_file()


def test_missing_filename_derives_extension_from_mime():
    path = spool_attachment(b"%PDF-1.4", None, "application/pdf")

    assert path is not None
    assert path.name.endswith(".pdf")


def test_unknown_mime_without_filename_falls_back_to_bin():
    path = spool_attachment(b"\x00\x01\x02", None, "application/x-not-a-real-mime")

    assert path is not None
    assert path.name.endswith(".bin")


@pytest.mark.parametrize(
    "hostile", ["../../etc/passwd", "/etc/passwd", "..", "C:\\windows\\evil.dll"]
)
def test_path_traversal_filenames_stay_inside_the_spool_dir(
    hostile, isolated_spool_dir
):
    path = spool_attachment(b"payload", hostile, "application/pdf")

    assert path is not None
    assert path.parent == isolated_spool_dir.resolve()
    assert "/" not in path.name and ".." not in path.name


def test_kill_switch_disables_spooling(monkeypatch):
    monkeypatch.setenv("OPEN_LLM_PROXY_SPOOL_ATTACHMENTS", "0")

    assert spool_attachment(b"data", "a.pdf", "application/pdf") is None


def test_empty_payload_is_not_spooled():
    assert spool_attachment(b"", "a.pdf", "application/pdf") is None


def test_write_failure_returns_none(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(attachment_spool.tempfile, "mkstemp", boom)

    assert spool_attachment(b"data", "a.pdf", "application/pdf") is None


def test_retention_pruning_removes_old_files_and_keeps_fresh_ones(
    isolated_spool_dir, monkeypatch
):
    monkeypatch.setenv("OPEN_LLM_PROXY_ATTACHMENT_RETENTION_DAYS", "7")
    fresh = spool_attachment(b"fresh bytes", "fresh.pdf", "application/pdf")
    stale = isolated_spool_dir / "deadbeef-stale.pdf"
    stale.write_bytes(b"stale")
    ancient = 8 * 86400
    os.utime(stale, (0, __import__("time").time() - ancient))

    spool_attachment(b"trigger a prune", "trigger.pdf", "application/pdf")

    assert not stale.exists()
    assert fresh.exists()


def test_zero_retention_disables_pruning(isolated_spool_dir, monkeypatch):
    monkeypatch.setenv("OPEN_LLM_PROXY_ATTACHMENT_RETENTION_DAYS", "0")
    isolated_spool_dir.mkdir(parents=True, exist_ok=True)
    stale = isolated_spool_dir / "deadbeef-stale.pdf"
    stale.write_bytes(b"stale")
    os.utime(stale, (0, 0))

    spool_attachment(b"trigger", "t.pdf", "application/pdf")

    assert stale.exists()


def test_invalid_retention_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("OPEN_LLM_PROXY_ATTACHMENT_RETENTION_DAYS", "not-a-number")

    assert spool_attachment(b"data", "a.pdf", "application/pdf") is not None


def test_pruning_never_raises_on_a_missing_directory(tmp_path):
    assert prune_spool_dir(tmp_path / "does-not-exist") == 0


def test_default_spool_dir_is_under_local_share(monkeypatch):
    monkeypatch.delenv("OPEN_LLM_PROXY_ATTACHMENT_DIR", raising=False)

    assert spool_dir() == Path.home() / ".local/share/open-llm-proxy/attachments"
