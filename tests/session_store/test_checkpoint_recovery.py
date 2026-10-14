from __future__ import annotations

import os
import shutil
import subprocess
import sys
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


def write(tools: FileTools, path: str, content: str, *, existing: bool = True) -> None:
    args = {"path": path, "content": content}
    if existing:
        args["expected_sha"] = tools.run(ToolCall("r", "read_file", {"path": path})).data["sha256"]
    assert tools.run(ToolCall("w", "write_file", args)).status is ToolStatus.OK


def open_turn(store: CheckpointStore, root: Path) -> FileTools:
    (root / "a.txt").write_text("A0", encoding="utf-8")
    tools = FileTools(root, journal=store)
    store.begin_turn("s1", root)
    write(tools, "a.txt", "A1")
    return tools


def turn_state(store: CheckpointStore) -> str:
    return store._connection.execute("SELECT state FROM ckpt_turns").fetchone()[0]


# -- #8 owner / liveness ---------------------------------------------------------------


def test_live_owner_turn_is_left_open_and_keeps_correct_post_image(tmp_path, root):
    db = tmp_path / "s.sqlite3"
    first = CheckpointStore(db)
    tools = open_turn(first, root)
    second = CheckpointStore(db, liveness=lambda pid, start: True)
    assert second.recover() == []
    assert turn_state(first) == "open"
    write(tools, "a.txt", "A2")  # the first session keeps working
    first.end_turn()
    assert first.undo_last("s1").ok and (root / "a.txt").read_text() == "A0"
    first.close()
    second.close()


def test_dead_owner_turn_is_closed(tmp_path, root):
    db = tmp_path / "s.sqlite3"
    first = CheckpointStore(db)
    open_turn(first, root)
    second = CheckpointStore(db, liveness=lambda pid, start: False)
    assert any("interrupted turn" in note for note in second.recover())
    assert turn_state(second) == "closed"
    first.close()
    second.close()


def test_liveness_receives_recorded_pid_and_start(tmp_path, root):
    db = tmp_path / "s.sqlite3"
    first = CheckpointStore(db)
    open_turn(first, root)
    seen: list[tuple[int, float | None]] = []
    second = CheckpointStore(db, liveness=lambda pid, start: seen.append((pid, start)) or True)
    second.recover()
    assert seen == [(first.owner[1], first.owner[2])]
    first.close()
    second.close()


def test_foreign_host_turn_is_left_open(tmp_path, root):
    db = tmp_path / "s.sqlite3"
    first = CheckpointStore(db)
    open_turn(first, root)
    first._connection.execute("UPDATE ckpt_turns SET owner_host='some-other-machine'")
    second = CheckpointStore(db, liveness=lambda pid, start: False)
    assert second.recover() == []
    assert turn_state(second) == "open"
    first.close()
    second.close()


def test_default_liveness_sees_this_process_alive_and_a_changed_start_as_dead():
    pid = os.getpid()
    start = ck.process_start(pid)
    assert ck.process_alive(pid, start)
    assert not ck.process_alive(pid, (start or 0.0) + 1000.0)  # pid reused by a younger process


def test_default_liveness_dead_pid():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    start = ck.process_start(child.pid)
    assert ck.process_alive(child.pid, start)
    child.kill()
    child.wait()
    assert not ck.process_alive(child.pid, start)


def test_legacy_db_without_owner_columns_is_migrated(tmp_path):
    import sqlite3

    db = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE ckpt_turns (turn_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,"
        " root TEXT NOT NULL, started_ns INTEGER NOT NULL, state TEXT NOT NULL DEFAULT 'open',"
        " commands INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute("INSERT INTO ckpt_turns(session_id, root, started_ns) VALUES ('s', ?, 1)", (str(tmp_path),))
    conn.commit()
    conn.close()
    store = CheckpointStore(db)
    assert any("interrupted turn" in note for note in store.recover())  # ownerless row: abandoned
    store.close()


# -- #2 recovery path safety -----------------------------------------------------------


def make_link(link: Path, target: Path) -> bool:
    try:
        if os.name == "nt":
            done = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False
            )
            return done.returncode == 0
        link.symlink_to(target, target_is_directory=True)
        return True
    except OSError:
        return False


def crash_on_second(real, calls):
    def wrapper(target, payload, mode):
        calls.append(target)
        if len(calls) == 2:
            raise KeyboardInterrupt
        real(target, payload, mode)

    return wrapper


@pytest.mark.parametrize("matching", ["pre", "post"])
def test_recovery_refuses_when_parent_became_a_link_outside(tmp_path, root, monkeypatch, matching):
    sub = root / "sub"
    sub.mkdir()
    (sub / "a.txt").write_text("A0", encoding="utf-8")
    (root / "b.txt").write_text("B0", encoding="utf-8")
    store = CheckpointStore(tmp_path / "s.sqlite3")
    tools = FileTools(root, journal=store)
    store.begin_turn("s1", root)
    write(tools, "sub/a.txt", "A1")
    write(tools, "b.txt", "B1")
    store.end_turn()
    real = ck.atomic_replace
    monkeypatch.setattr(ck, "atomic_replace", crash_on_second(real, []))
    with pytest.raises(KeyboardInterrupt):
        store.undo_last("s1")
    monkeypatch.setattr(ck, "atomic_replace", real)
    store.close()

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.txt").write_text("A1" if matching == "post" else "A0", encoding="utf-8")
    shutil.rmtree(sub)
    if not make_link(sub, outside):
        pytest.skip("cannot create a directory link here")
    before = (outside / "a.txt").read_bytes()

    fresh = CheckpointStore(tmp_path / "s.sqlite3", liveness=lambda pid, start: False)
    notes = fresh.recover()
    assert (outside / "a.txt").read_bytes() == before
    assert sorted(p.name for p in outside.iterdir()) == ["a.txt"]
    assert any("pending" in note for note in notes)
    state = fresh._connection.execute("SELECT state FROM ckpt_undo").fetchone()[0]
    assert state == "restoring"
    assert not fresh.undo_last("s1").ok  # new undo refused while recovery is pending
    fresh.close()


def test_recovery_refuses_protected_target(tmp_path, root, monkeypatch):
    (root / "a.txt").write_text("A0", encoding="utf-8")
    (root / "b.txt").write_text("B0", encoding="utf-8")
    store = CheckpointStore(tmp_path / "s.sqlite3")
    tools = FileTools(root, journal=store)
    store.begin_turn("s1", root)
    write(tools, "a.txt", "A1")
    write(tools, "b.txt", "B1")
    store.end_turn()
    real = ck.atomic_replace
    monkeypatch.setattr(ck, "atomic_replace", crash_on_second(real, []))
    with pytest.raises(KeyboardInterrupt):
        store.undo_last("s1")
    monkeypatch.setattr(ck, "atomic_replace", real)
    store._connection.execute("UPDATE ckpt_files SET rel='.git/hooks/b.txt', key='.git/hooks/b.txt' WHERE rel='b.txt'")
    store.close()
    fresh = CheckpointStore(tmp_path / "s.sqlite3")
    notes = fresh.recover()
    fresh.close()
    assert any("pending" in note for note in notes)
    assert not (root / ".git").exists()
