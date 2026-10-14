"""expected_sha matching as real local models produce it (eval/TUI traces, 2026-10-14)."""

from __future__ import annotations

from pathlib import Path

from ironclad.contracts import ToolCall, ToolStatus
from ironclad.tool_runtime.files import FileTools, hash_matches, sha256_hex

DIGEST = sha256_hex(b"x = 1\n")


def test_prefix_rules() -> None:
    assert hash_matches(DIGEST, DIGEST) and hash_matches(DIGEST[:16], DIGEST)
    assert hash_matches("sha256:" + DIGEST[:16].upper(), DIGEST)
    invented = DIGEST[:16] + "0" * 48  # model "completed" the 16 shown digits
    assert hash_matches(invented, DIGEST)
    assert not hash_matches(DIGEST[:15], DIGEST)  # shorter than what read_file shows
    assert not hash_matches("", DIGEST)
    assert not hash_matches(("f" if DIGEST[0] != "f" else "e") + DIGEST[1:], DIGEST)


def _tools(tmp_path: Path) -> FileTools:
    (tmp_path / "a.py").write_bytes(b"x = 1\n")
    return FileTools(tmp_path)


def test_invented_full_sha_with_the_shown_prefix_edits(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    read = tools.run(ToolCall("1", "read_file", {"path": "a.py"}))
    shown = read.output.split("sha256:", 1)[1][:16]
    edit = tools.run(ToolCall("2", "edit_file", {"path": "a.py", "old": "1", "new": "2", "expected_sha": shown + "ab" * 24}))
    assert edit.status is ToolStatus.OK, edit.output
    assert (tmp_path / "a.py").read_bytes() == b"x = 2\n"


def test_wrong_sha_after_a_read_says_which_sha_to_use(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    tools.run(ToolCall("1", "read_file", {"path": "a.py"}))
    bad = tools.run(ToolCall("2", "edit_file", {"path": "a.py", "old": "1", "new": "2", "expected_sha": "0" * 16}))
    assert bad.status is ToolStatus.INVALID and DIGEST[:16] in bad.output
    missing = tools.run(ToolCall("3", "write_file", {"path": "a.py", "content": "x = 3\n"}))
    assert missing.status is ToolStatus.INVALID and DIGEST[:16] in missing.output


def test_sha_is_never_revealed_without_a_read_of_the_current_bytes(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    bad = tools.run(ToolCall("1", "edit_file", {"path": "a.py", "old": "1", "new": "2", "expected_sha": "0" * 16}))
    assert bad.status is ToolStatus.INVALID and DIGEST[:16] not in bad.output
    tools.run(ToolCall("2", "read_file", {"path": "a.py"}))
    (tmp_path / "a.py").write_bytes(b"x = 9\n")  # someone else changed it after the read
    stale = tools.run(ToolCall("3", "edit_file", {"path": "a.py", "old": "9", "new": "2", "expected_sha": DIGEST[:16]}))
    assert stale.status is ToolStatus.INVALID
    assert sha256_hex(b"x = 9\n")[:16] not in stale.output and "read it first" in stale.output
