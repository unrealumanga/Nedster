from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from ironclad.contracts import ToolCall, ToolStatus
from ironclad.session_store import checkpoints as ck
from ironclad.session_store.checkpoints import CheckpointStore
from ironclad.tool_runtime.files import FileTools


@pytest.fixture
def root(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    return project


@pytest.fixture
def store(tmp_path: Path):
    s = CheckpointStore(tmp_path / "state" / "s.sqlite3")
    yield s
    s.close()


def sha_of(tools: FileTools, rel: str) -> str:
    return tools.run(ToolCall("r", "read_file", {"path": rel})).data["sha256"]


def write(tools: FileTools, path: str, content: str, *, existing: bool = True):
    args = {"path": path, "content": content}
    if existing:
        args["expected_sha"] = sha_of(tools, path)
    result = tools.run(ToolCall("w", "write_file", args))
    assert result.status is ToolStatus.OK, result.output
    return result


def turn(store: CheckpointStore, root: Path, sid: str = "s1") -> None:
    store.begin_turn(sid, root)


def crash_on_second(real, calls):
    def wrapper(target, payload, mode):
        calls.append(target)
        if len(calls) == 2:
            raise KeyboardInterrupt  # simulated process death mid-undo (not an OSError)
        real(target, payload, mode)

    return wrapper


def two_file_turn(store: CheckpointStore, root: Path) -> None:
    (root / "a.txt").write_text("A0", encoding="utf-8")
    (root / "b.txt").write_text("B0", encoding="utf-8")
    tools = FileTools(root, journal=store)
    turn(store, root)
    write(tools, "a.txt", "A1")
    write(tools, "b.txt", "B1")
    store.end_turn()


def test_multi_file_undo_restores_edit_and_removes_created(store, root):
    (root / "a.txt").write_text("A0", encoding="utf-8")
    (root / "b.txt").write_text("B0", encoding="utf-8")
    tools = FileTools(root, journal=store)
    turn(store, root)
    write(tools, "a.txt", "A1")
    write(tools, "a.txt", "A2")  # second mutation: pre-image must stay A0
    write(tools, "b.txt", "B1")
    write(tools, "new/deep/c.txt", "C", existing=False)
    store.end_turn()
    report = store.undo_last("s1")
    assert report.ok, report.message
    assert (root / "a.txt").read_text() == "A0" and (root / "b.txt").read_text() == "B0"
    assert not (root / "new").exists()  # created file and its created parents are gone
    assert sorted(report.restored) == ["a.txt", "b.txt"] and report.removed == ["new/deep/c.txt"]
    assert not report.commands_not_covered and "NOT covered" not in report.message


def test_deleted_file_is_restored(store, root):
    target = root / "gone.txt"
    target.write_bytes(b"bytes\r\n")
    turn(store, root)
    store.record_pre_image("gone.txt", target)
    target.unlink()
    store.end_turn()
    assert store.undo_last("s1").ok
    assert target.read_bytes() == b"bytes\r\n"


def test_user_edit_after_turn_refuses_and_restores_nothing(store, root):
    two_file_turn(store, root)
    (root / "b.txt").write_text("user edit", encoding="utf-8")
    report = store.undo_last("s1")
    assert not report.ok and [c.path for c in report.conflicts] == ["b.txt"]
    assert (root / "a.txt").read_text() == "A1" and (root / "b.txt").read_text() == "user edit"
    assert "nothing was changed" in report.message


def test_commands_are_reported_as_not_covered(store, root):
    (root / "a.txt").write_text("A0", encoding="utf-8")
    tools = FileTools(root, journal=store)
    turn(store, root)
    write(tools, "a.txt", "A1")
    store.note_command()
    store.end_turn()
    report = store.undo_last("s1")
    assert report.ok and report.commands_not_covered
    assert "NOT covered" in report.message


def test_failed_partial_write_is_journaled_and_undoable(store, root, monkeypatch):
    (root / "a.txt").write_text("A0", encoding="utf-8")
    tools = FileTools(root, journal=store)
    turn(store, root)
    sha = sha_of(tools, "a.txt")

    def partial(target: Path, payload: bytes, mode):
        target.write_bytes(payload[:1])  # torn write then failure
        raise OSError("disk full")

    monkeypatch.setattr("ironclad.tool_runtime.files.atomic_replace", partial)
    result = tools.run(ToolCall("w", "write_file", {"path": "a.txt", "content": "A1", "expected_sha": sha}))
    assert result.status is ToolStatus.FAILED
    store.end_turn()
    assert (root / "a.txt").read_bytes() == b"A"
    assert store.undo_last("s1").ok and (root / "a.txt").read_text() == "A0"


def test_journal_failure_refuses_the_mutation(root, tmp_path):
    store = CheckpointStore(tmp_path / "j.sqlite3")
    tools = FileTools(root, journal=store)  # no turn declared -> cannot journal
    result = tools.run(ToolCall("w", "write_file", {"path": "x.txt", "content": "x"}))
    assert result.status is ToolStatus.FAILED and not (root / "x.txt").exists()
    store.close()


def test_interrupted_restore_rolls_forward(store, root, monkeypatch):
    two_file_turn(store, root)
    real = ck.atomic_replace
    monkeypatch.setattr(ck, "atomic_replace", crash_on_second(real, []))
    with pytest.raises(KeyboardInterrupt):
        store.undo_last("s1")
    monkeypatch.setattr(ck, "atomic_replace", real)
    assert (root / "a.txt").read_text() == "A0" and (root / "b.txt").read_text() == "B1"
    fresh = CheckpointStore(store.path)
    notes = fresh.recover()
    fresh.close()
    assert any("completed" in note for note in notes)
    assert (root / "a.txt").read_text() == "A0" and (root / "b.txt").read_text() == "B0"


def test_interrupted_restore_rolls_back_when_a_file_diverged(store, root, monkeypatch):
    two_file_turn(store, root)
    real = ck.atomic_replace
    monkeypatch.setattr(ck, "atomic_replace", crash_on_second(real, []))
    with pytest.raises(KeyboardInterrupt):
        store.undo_last("s1")
    monkeypatch.setattr(ck, "atomic_replace", real)
    (root / "b.txt").write_text("user edit", encoding="utf-8")
    fresh = CheckpointStore(store.path)
    notes = fresh.recover()
    fresh.close()
    assert any("rolled back" in note for note in notes)
    assert (root / "a.txt").read_text() == "A1" and (root / "b.txt").read_text() == "user edit"


def test_crash_leaves_open_turn_which_recover_closes(store, root):
    (root / "a.txt").write_text("A0", encoding="utf-8")
    tools = FileTools(root, journal=store)
    turn(store, root)
    write(tools, "a.txt", "A1")  # process dies before end_turn
    fresh = CheckpointStore(store.path, liveness=lambda pid, start: False)  # owner process is gone
    assert any("interrupted turn" in note for note in fresh.recover())
    assert fresh.undo_last("s1").ok and (root / "a.txt").read_text() == "A0"
    fresh.close()


def test_protected_path_refused(store, root):
    (root / ".nedster").mkdir()
    protected = root / ".nedster" / "state.json"
    protected.write_text("p0", encoding="utf-8")
    turn(store, root)
    store.record_pre_image(".nedster/state.json", protected)
    protected.write_text("p1", encoding="utf-8")
    store.end_turn()
    report = store.undo_last("s1")
    assert not report.ok and "protected" in report.conflicts[0].reason
    assert protected.read_text() == "p1"


def test_path_escaping_root_refused(store, root, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("o0", encoding="utf-8")
    turn(store, root)
    store.record_pre_image("../outside.txt", outside)
    outside.write_text("o1", encoding="utf-8")
    store.end_turn()
    report = store.undo_last("s1")
    assert not report.ok and outside.read_text() == "o1"


@pytest.mark.skipif(os.name != "nt", reason="case-insensitive filesystem semantics")
def test_windows_path_case_is_one_journal_entry(store, root):
    (root / "Readme.md").write_text("R0", encoding="utf-8")
    turn(store, root)
    store.record_pre_image("Readme.md", root / "Readme.md")
    (root / "Readme.md").write_text("R1", encoding="utf-8")
    store.record_pre_image("README.MD", root / "README.MD")  # same file, other spelling: must not re-capture
    (root / "Readme.md").write_text("R2", encoding="utf-8")
    store.end_turn()
    assert store.undo_last("s1").ok and (root / "Readme.md").read_text() == "R0"


def test_undone_turn_cannot_be_undone_twice_and_hook_fires(store, root):
    (root / "a.txt").write_text("A0", encoding="utf-8")
    seen: list[list[str]] = []
    store.on_undone.append(seen.append)
    tools = FileTools(root, journal=store)
    turn(store, root)
    write(tools, "a.txt", "A1")
    store.end_turn()
    assert store.undo_last("s1").ok and seen == [["a.txt"]]
    assert not store.undo_last("s1").ok


def test_turns_without_mutations_leave_no_rows(store, root):
    turn(store, root)
    store.end_turn()
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM ckpt_turns").fetchone()[0] == 0
