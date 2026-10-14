from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ironclad.contracts import ToolCall, ToolStatus
from ironclad.tool_runtime.files import EDIT_SCHEMA, FileTools


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id="m1", name=name, arguments=arguments)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "a.txt").write_bytes(b"one\ntwo\nthree\n")
    return project


@pytest.fixture
def tools(root: Path) -> FileTools:
    return FileTools(root)


def read_sha(tools: FileTools, path: str = "a.txt") -> str:
    return tools.run(call("read_file", path=path)).data["sha256"]


def test_edit_result_states_new_sha_and_chains_without_reread(tools: FileTools, root: Path):
    first = tools.run(call("edit_file", path="a.txt", old="one", new="1", expected_sha=read_sha(tools)))
    assert first.status is ToolStatus.OK, first.output
    expected = sha(b"1\ntwo\nthree\n")
    assert first.data["sha"] == expected[:16] and first.data["sha256"] == expected
    assert f"sha256:{expected[:16]}" in first.output
    second = tools.run(call("edit_file", path="a.txt", old="two", new="2", expected_sha=first.data["sha"]))
    assert second.status is ToolStatus.OK, second.output
    assert (root / "a.txt").read_bytes() == b"1\n2\nthree\n"


def test_write_result_states_new_sha_and_chains(tools: FileTools, root: Path):
    wrote = tools.run(call("write_file", path="a.txt", content="x\n", expected_sha=read_sha(tools)))
    assert wrote.data["sha"] == sha(b"x\n")[:16] and f"sha256:{wrote.data['sha']}" in wrote.output
    assert tools.read_hashes["a.txt"] == sha(b"x\n")
    again = tools.run(call("edit_file", path="a.txt", old="x", new="y", expected_sha=wrote.data["sha"]))
    assert again.status is ToolStatus.OK


def test_multi_edit_applies_in_order_in_one_write(tools: FileTools, root: Path):
    result = tools.run(
        call(
            "edit_file",
            path="a.txt",
            expected_sha=read_sha(tools),
            edits=[{"old": "one", "new": "uno"}, {"old": "uno\ntwo", "new": "UNO-TWO"}, {"old": "three", "new": "3"}],
        )
    )
    assert result.status is ToolStatus.OK, result.output
    assert (root / "a.txt").read_bytes() == b"UNO-TWO\n3\n"
    assert result.mutated_paths == ("a.txt",)
    assert result.data["sha"] == sha(b"UNO-TWO\n3\n")[:16]


def test_multi_edit_failure_writes_nothing_and_names_index(tools: FileTools, root: Path):
    before = (root / "a.txt").read_bytes()
    result = tools.run(
        call(
            "edit_file",
            path="a.txt",
            expected_sha=read_sha(tools),
            edits=[{"old": "one", "new": "1"}, {"old": "zzz", "new": "x"}],
        )
    )
    assert result.status is ToolStatus.INVALID
    assert "edits[1]" in result.output and "occurs 0 time(s)" in result.output
    assert (root / "a.txt").read_bytes() == before


def test_multi_edit_each_old_must_match_exactly_once(tools: FileTools, root: Path):
    (root / "d.txt").write_bytes(b"x\nx\n")
    result = tools.run(
        call("edit_file", path="d.txt", expected_sha=read_sha(tools, "d.txt"), edits=[{"old": "x", "new": "y"}])
    )
    assert result.status is ToolStatus.INVALID and "edits[0]" in result.output and "occurs 2" in result.output
    assert (root / "d.txt").read_bytes() == b"x\nx\n"


def test_second_edit_sees_text_left_by_first(tools: FileTools, root: Path):
    result = tools.run(
        call(
            "edit_file",
            path="a.txt",
            expected_sha=read_sha(tools),
            edits=[{"old": "one", "new": "two"}, {"old": "two", "new": "2"}],
        )
    )
    assert result.status is ToolStatus.INVALID and "edits[1]" in result.output and "occurs 2" in result.output


@pytest.mark.parametrize(
    "extra",
    [
        {},  # neither form
        {"old": "one", "new": "1", "edits": [{"old": "two", "new": "2"}]},  # both forms
        {"edits": []},
        {"edits": [{"old": "one", "new": "1"}] * 51},
        {"edits": [{"old": "", "new": "1"}]},
        {"old": "one"},
    ],
)
def test_bad_edit_shapes_are_invalid_and_leave_file(tools: FileTools, root: Path, extra: dict):
    before = (root / "a.txt").read_bytes()
    result = tools.run(call("edit_file", path="a.txt", expected_sha=read_sha(tools), **extra))
    assert result.status is ToolStatus.INVALID
    assert (root / "a.txt").read_bytes() == before


def test_edits_items_are_strict(tools: FileTools, root: Path):
    sha_ = read_sha(tools)
    extra = tools.run(call("edit_file", path="a.txt", expected_sha=sha_, edits=[{"old": "one", "new": "1", "x": 1}]))
    assert extra.status is ToolStatus.INVALID and "unexpected field" in extra.output
    missing = tools.run(call("edit_file", path="a.txt", expected_sha=sha_, edits=[{"old": "one"}]))
    assert missing.status is ToolStatus.INVALID
    unknown = tools.run(call("edit_file", path="a.txt", expected_sha=sha_, edits=[{"old": "one", "new": "1"}], bogus=1))
    assert unknown.status is ToolStatus.INVALID
    assert EDIT_SCHEMA["additionalProperties"] is False and "oneOf" in EDIT_SCHEMA


def test_multi_edit_keeps_expected_sha_semantics(tools: FileTools, root: Path):
    read_sha(tools)
    result = tools.run(call("edit_file", path="a.txt", expected_sha="0" * 64, edits=[{"old": "one", "new": "1"}]))
    # unchanged since the model's own read: naming the sha it already saw reveals nothing new
    assert result.status is ToolStatus.INVALID and "unchanged since your read" in result.output
    assert sha(b"one\ntwo\nthree\n")[:16] in result.output


def test_stale_after_own_edit_reveals_current_sha(tools: FileTools, root: Path):
    old_sha = read_sha(tools)
    first = tools.run(call("edit_file", path="a.txt", old="one", new="1", expected_sha=old_sha))
    stale = tools.run(call("edit_file", path="a.txt", old="two", new="2", expected_sha=old_sha))
    assert stale.status is ToolStatus.INVALID
    assert "by your own edit at" in stale.output and f"current sha256:{first.data['sha']}" in stale.output
    assert (root / "a.txt").read_bytes() == b"1\ntwo\nthree\n"


def test_stale_after_foreign_change_does_not_reveal_sha(tools: FileTools, root: Path):
    old_sha = read_sha(tools)
    tools.run(call("edit_file", path="a.txt", old="one", new="1", expected_sha=old_sha))
    foreign = b"someone\nelse\n"
    (root / "a.txt").write_bytes(foreign)
    stale = tools.run(call("edit_file", path="a.txt", old="someone", new="x", expected_sha=old_sha))
    assert stale.status is ToolStatus.INVALID and "read it first" in stale.output
    assert sha(foreign)[:16] not in stale.output and "own edit" not in stale.output
    stale_write = tools.run(call("write_file", path="a.txt", content="x", expected_sha=old_sha))
    assert sha(foreign)[:16] not in stale_write.output


def test_untouched_file_changed_externally_does_not_reveal_sha(tools: FileTools, root: Path):
    old_sha = read_sha(tools)
    (root / "a.txt").write_bytes(b"external\n")
    stale = tools.run(call("edit_file", path="a.txt", old="external", new="x", expected_sha=old_sha))
    assert "read it first" in stale.output and sha(b"external\n")[:16] not in stale.output


def test_invalidate_forgets_own_write_record(tools: FileTools, root: Path):
    old_sha = read_sha(tools)
    tools.run(call("edit_file", path="a.txt", old="one", new="1", expected_sha=old_sha))
    tools.invalidate_read_hashes(["a.txt"])
    stale = tools.run(call("edit_file", path="a.txt", old="two", new="2", expected_sha=old_sha))
    assert "read it first" in stale.output and "current sha" not in stale.output


def test_multi_edit_records_one_pre_image_before_mutation(root: Path):
    events: list[str] = []

    class Journal:
        import threading

        mutation_lock = threading.RLock()

        def record_pre_image(self, rel: str, path: Path) -> None:
            events.append(f"pre:{rel}:{path.read_bytes()!r}")

    tools = FileTools(root, journal=Journal())
    result = tools.run(
        call(
            "edit_file",
            path="a.txt",
            expected_sha=read_sha(tools),
            edits=[{"old": "one", "new": "1"}, {"old": "two", "new": "2"}],
        )
    )
    assert result.status is ToolStatus.OK
    assert events == ["pre:a.txt:b'one\\ntwo\\nthree\\n'"]


def test_failed_multi_edit_records_no_pre_image(root: Path):
    events: list[str] = []

    class Journal:
        import threading

        mutation_lock = threading.RLock()

        def record_pre_image(self, rel: str, path: Path) -> None:
            events.append(rel)

    tools = FileTools(root, journal=Journal())
    result = tools.run(
        call("edit_file", path="a.txt", expected_sha=read_sha(tools), edits=[{"old": "nope", "new": "x"}])
    )
    assert result.status is ToolStatus.INVALID and events == []


def test_multi_edit_protected_path_and_crlf(tools: FileTools, root: Path):
    protected = tools.run(call("edit_file", path=".git/config", expected_sha="0" * 16, edits=[{"old": "a", "new": "b"}]))
    assert protected.status is ToolStatus.INVALID and "protected" in protected.output
    (root / "crlf.txt").write_bytes(b"a\r\nb\r\nc\r\n")
    result = tools.run(
        call(
            "edit_file",
            path="crlf.txt",
            expected_sha=read_sha(tools, "crlf.txt"),
            edits=[{"old": "a", "new": "A"}, {"old": "c", "new": "C"}],
        )
    )
    assert result.status is ToolStatus.OK
    assert (root / "crlf.txt").read_bytes() == b"A\r\nb\r\nC\r\n"
