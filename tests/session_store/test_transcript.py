"""W2-RESUME: canonical display transcript, activity locks, interrupted-call repair, prune safety."""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest

from ironclad.contracts import ChatMessage, ToolCall
from ironclad.session_store.sqlite_store import (
    INTERRUPTED_TEXT,
    LOCK_TTL_S,
    SCHEMA_VERSION,
    SqliteSessionStore,
    make_owner,
    normalize_project_root,
)


@pytest.fixture
def store(tmp_path: Path):
    with SqliteSessionStore(tmp_path / "sessions.sqlite3") as session_store:
        yield session_store


# --- invariant 1: display transcript is separate from the model history -----------------------


def test_transcript_round_trip_keeps_order_roles_and_meta(store: SqliteSessionStore):
    session = store.create(str(Path.cwd()))
    assert store.append_transcript(session, "user", "hi", {"source": "cli"}) == 1
    store.append_transcript(session, "assistant", "hello")
    store.append_transcript(session, "tool", "file.py", {"tool": "read_file", "status": "ok"})

    entries = store.transcript(session)
    assert [entry["role"] for entry in entries] == ["user", "assistant", "tool"]
    assert [entry["seq"] for entry in entries] == [1, 2, 3]
    assert entries[0]["meta"] == {"source": "cli"}
    assert entries[1]["meta"] == {} and entries[1]["text"] == "hello"
    assert entries[2]["meta"]["tool"] == "read_file"
    assert all(entry["created_ns"] > 0 for entry in entries)
    assert store.transcript_count(session) == 3


def test_transcript_limit_returns_only_the_newest_entries_still_oldest_first(store: SqliteSessionStore):
    session = store.create(str(Path.cwd()))
    for index in range(5):
        store.append_transcript(session, "user", f"msg-{index}")

    newest = store.transcript(session, limit=2)
    assert [entry["text"] for entry in newest] == ["msg-3", "msg-4"]
    assert len(store.transcript(session, limit=0)) == 5
    assert store.transcript(session, limit=-1)[0]["text"] == "msg-0"


def test_transcript_never_leaks_between_sessions_or_projects(store: SqliteSessionStore):
    one, two = store.create("/one"), store.create("/two")
    store.append_transcript(one, "user", "only one")
    store.append_transcript(two, "user", "only two")

    assert [entry["text"] for entry in store.transcript(one)] == ["only one"]
    assert [entry["text"] for entry in store.transcript(two)] == ["only two"]
    assert store.transcript("missing-session") == []


def test_replace_messages_rewrites_model_history_but_never_the_transcript(tmp_path: Path):
    path = tmp_path / "sessions.sqlite3"
    with SqliteSessionStore(path) as store:
        session = store.create(str(tmp_path))
        store.append_transcript(session, "user", "question")
        store.append_transcript(session, "assistant", "full unmasked answer")
        history = [
            ChatMessage(role="user", content="question"),
            ChatMessage(
                role="assistant",
                content="running a tool",
                tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": "ls"})],
            ),
            ChatMessage(role="tool", content="[output #1 elided: bash ok ls]", tool_call_id="c1", name="bash"),
        ]
        for message in history:
            store.append_message(session, message)

        store.replace_messages(session, [ChatMessage(role="user", content="[compaction summary #1: gone]")])

        assert [m.content for m in store.messages(session)] == ["[compaction summary #1: gone]"]
        assert [entry["text"] for entry in store.transcript(session)] == [
            "question",
            "full unmasked answer",
        ]


def test_transcript_persists_across_reopen(tmp_path: Path):
    path = tmp_path / "sessions.sqlite3"
    with SqliteSessionStore(path) as store:
        session = store.create(str(tmp_path), title="t")
        store.append_transcript(session, "user", "kept", {"k": [1, 2]})
    with SqliteSessionStore(path) as store:
        entries = store.transcript(session)
        assert entries[0]["text"] == "kept"
        assert entries[0]["meta"] == {"k": [1, 2]}
        assert store.transcript_count(session) == 1


def test_transcript_rejects_bad_roles_and_text(store: SqliteSessionStore):
    session = store.create(str(Path.cwd()))
    with pytest.raises(ValueError):
        store.append_transcript(session, "  ", "text")
    with pytest.raises(TypeError):
        store.append_transcript(session, "user", b"bytes")  # type: ignore[arg-type]
    store.append_transcript(session, "user", "ok", {"path": Path("a")})  # meta is display-only: str() fallback
    assert store.transcript(session)[0]["meta"]["path"] == "a"


def test_fresh_file_has_the_transcript_and_lock_tables(tmp_path: Path):
    with SqliteSessionStore(tmp_path / "fresh.sqlite3") as store:
        assert store.schema_version() == SCHEMA_VERSION
    with closing(sqlite3.connect(tmp_path / "fresh.sqlite3")) as connection:
        tables = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert {"transcript", "session_locks"} <= tables


# --- invariant 3: sessions are bound to the canonical project root ---------------------------


def test_sessions_bind_to_the_canonical_project_root(tmp_path: Path):
    root = tmp_path / "Proj"
    root.mkdir()
    other = tmp_path / "Other"
    other.mkdir()
    with SqliteSessionStore(tmp_path / "sessions.sqlite3") as store:
        session = store.create(str(root))
        store.create(str(other))

        assert [entry["id"] for entry in store.list_sessions(str(root))] == [session]
        assert store.list_sessions(str(root) + os.sep)[0]["id"] == session
        assert store.list_sessions(str(root) + "/")[0]["id"] == session
        assert store.list_sessions(str(other)) != [session]
        assert store.get_session(session)["project_root"] == normalize_project_root(str(root))

        try:
            relative = os.path.relpath(root)
        except ValueError:  # different drive (Windows): a relative path cannot name this root
            relative = None
        if relative is not None:
            assert store.list_sessions(relative)[0]["id"] == session
        if sys.platform == "win32":  # case-normalised on Windows only
            assert store.list_sessions(str(root).upper())[0]["id"] == session


# --- invariant 6: interrupted tool calls are repaired on load --------------------------------


def _killed_history() -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content="go"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="c1", name="bash", arguments={"command": "pytest"}),
                ToolCall(id="c2", name="bash", arguments={"command": "ls"}),
            ],
        ),
        ChatMessage(role="tool", content="[ok] first", tool_call_id="c1", name="bash"),
        ChatMessage(role="user", content="and then?"),
    ]


def test_repair_interrupted_answers_unpaired_calls_once_and_matches_the_engine(store: SqliteSessionStore):
    from ironclad.engine.loop import INTERRUPTED, repair_history

    session = store.create(str(Path.cwd()))
    for message in _killed_history():
        store.append_message(session, message)

    assert store.repair_interrupted(session) == 1
    repaired = store.messages(session)
    assert [m.role for m in repaired] == ["user", "assistant", "tool", "tool", "user"]
    assert repaired[3].tool_call_id == "c2" and repaired[3].name == "bash"
    assert repaired[3].content == INTERRUPTED_TEXT == INTERRUPTED
    assert repaired == repair_history(_killed_history())  # same placement as the engine's loader
    assert store.repair_interrupted(session) == 0  # idempotent


def test_repair_interrupted_mid_history_rewrites_via_replace_messages(store: SqliteSessionStore):
    history = [
        ChatMessage(role="user", content="go"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": "pytest"})],
        ),
        ChatMessage(role="assistant", content="stopped early"),  # no result ever arrived
        ChatMessage(role="user", content="continue"),
    ]
    session = store.create(str(Path.cwd()))
    for message in history:
        store.append_message(session, message)

    assert store.repair_interrupted(session) == 1
    roles = [(m.role, m.tool_call_id) for m in store.messages(session)]
    assert roles == [
        ("user", None),
        ("assistant", None),
        ("tool", "c1"),
        ("assistant", None),
        ("user", None),
    ]
    assert store.messages(session)[2].content == INTERRUPTED_TEXT


def test_repair_interrupted_is_a_no_op_for_a_complete_history(store: SqliteSessionStore):
    session = store.create(str(Path.cwd()))
    store.append_message(
        session,
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="bash", arguments={})],
        ),
    )
    store.append_message(session, ChatMessage(role="tool", content="[ok]", tool_call_id="c1", name="bash"))
    assert store.repair_interrupted(session) == 0
    assert len(store.messages(session)) == 2
    assert store.repair_interrupted("unknown-session") == 0


# --- invariant 4: an active (locked) session cannot be deleted --------------------------------


def test_lock_blocks_a_second_holder_and_delete_until_released(store: SqliteSessionStore):
    session = store.create(str(Path.cwd()))
    mine, theirs = make_owner("tui"), make_owner("cli")

    assert store.acquire_lock(session, mine) is True
    assert store.is_active(session) is True
    assert store.acquire_lock(session, theirs) is False  # live foreign holder wins
    assert store.delete_session(session) is False  # refusal, nothing deleted
    assert store.get_session(session) is not None

    assert store.refresh_lock(session, mine) is True
    assert store.refresh_lock(session, theirs) is False  # a stranger cannot heartbeat our lock

    store.release_lock(session, mine)
    assert store.is_active(session) is False
    assert store.acquire_lock(session, theirs) is True  # takeover once nobody holds it
    store.release_lock(session, theirs)
    assert store.delete_session(session, force=False) is True
    assert store.get_session(session) is None


def test_stale_lock_is_taken_over_and_delete_is_allowed_after_the_ttl(store: SqliteSessionStore):
    session = store.create(str(Path.cwd()))
    crashed = make_owner("tui")
    assert store.acquire_lock(session, crashed) is True
    assert store.is_active(session) is True

    with store._connection:  # the holder died: no heartbeat for longer than LOCK_TTL_S
        store._connection.execute(
            "UPDATE session_locks SET heartbeat_ns = ? WHERE session_id = ?",
            (time.time_ns() - int(LOCK_TTL_S * 2 * 1e9), session),
        )

    assert store.is_active(session) is False
    assert store.delete_session(session) is True

    # and a fresh process can take the session over
    survivor = store.create(str(Path.cwd()))
    assert store.acquire_lock(survivor, crashed) is True
    with store._connection:
        store._connection.execute(
            "UPDATE session_locks SET heartbeat_ns = ? WHERE session_id = ?",
            (time.time_ns() - int(LOCK_TTL_S * 2 * 1e9), survivor),
        )
    assert store.acquire_lock(survivor, make_owner("tui2")) is True
    assert store.is_active(survivor) is True


def test_lock_info_reports_owner_pid_and_activity(store: SqliteSessionStore):
    session = store.create(str(Path.cwd()))
    assert store.lock_info(session) is None
    owner = make_owner("cli")
    store.acquire_lock(session, owner)
    info = store.lock_info(session)
    assert info["owner"] == owner and info["pid"] == os.getpid()
    assert info["active"] is True and info["heartbeat_ns"] >= info["acquired_ns"] > 0


def test_acquire_lock_requires_an_existing_session(store: SqliteSessionStore):
    with pytest.raises(KeyError):
        store.acquire_lock("nope", make_owner("tui"))
    assert store.refresh_lock("nope", make_owner("tui")) is False
    assert store.delete_session("nope") is True  # idempotent


def test_delete_session_removes_every_row_of_that_session_only(tmp_path: Path):
    path = tmp_path / "sessions.sqlite3"
    with SqliteSessionStore(path) as store:
        gone, kept = store.create(str(tmp_path)), store.create(str(tmp_path))
        store.append_transcript(gone, "user", "bye")
        store.append_message(gone, ChatMessage(role="user", content="bye"))
        store.put_output(gone, "bash", "c1", "payload")
        store.set_state(gone, "task_state", {"a": 1})
        store.acquire_lock(gone, make_owner("tui"))
        store.append_transcript(kept, "user", "stay")
        store.append_message(kept, ChatMessage(role="user", content="stay"))

        assert store.delete_session(gone, force=True) is True
        assert store.get_session(gone) is None
        assert store.transcript(gone) == [] and store.messages(gone) == []
        assert store.state_keys(gone) == [] and store.lock_info(gone) is None
        with pytest.raises(KeyError):
            store.get_output(gone, 1)
        assert [e["text"] for e in store.transcript(kept)] == ["stay"]
        assert len(store.messages(kept)) == 1


# --- invariant 5: prune deletes only unreferenced, expired outputs ---------------------------


def _old_outputs(store: SqliteSessionStore, session: str) -> dict[str, int]:
    handles = {
        "stub": store.put_output(session, "bash", "c1", "elided by masking"),
        "truncated": store.put_output(session, "bash", "c2", "truncated by tool_message"),
        "summary": store.put_output(session, "bash", "c3", "dropped with a group"),
        "state": store.put_output(session, "bash", "c4", "kept in task state"),
        "unreferenced": store.put_output(session, "bash", "c5", "nobody points here"),
        "recent": store.put_output(session, "bash", "c6", "fresh"),
    }
    with store._connection:
        store._connection.execute("UPDATE outputs SET created_ns = 1 WHERE session_id = ?", (session,))
        store._connection.execute(
            "UPDATE outputs SET created_ns = ? WHERE session_id = ? AND handle = ?",
            (time.time_ns(), session, handles["recent"]),
        )
    return handles


def test_prune_removes_only_old_unreferenced_outputs_of_this_project(tmp_path: Path):
    with SqliteSessionStore(tmp_path / "sessions.sqlite3") as store:
        session = store.create(str(tmp_path))
        other_root = tmp_path / "other"
        other_root.mkdir()
        foreign = store.create(str(other_root))
        handles = _old_outputs(store, session)
        foreign_handle = store.put_output(foreign, "bash", "x", "other project")

        store.append_message(
            session,
            ChatMessage(role="assistant", content="", tool_calls=[ToolCall(id="c3", name="show_output", arguments={"handle": handles["summary"]})]),
        )
        store.append_message(
            session,
            ChatMessage(role="tool", content=f"[output #{handles['stub']} elided: bash ok ...]", tool_call_id="c3", name="bash"),
        )
        store.append_message(
            session,
            ChatMessage(role="tool", content=f"[ok] full text: show_output(handle={handles['truncated']})", tool_call_id="c4", name="bash"),
        )
        store.set_state(session, "task_state", {"summary": {"handles": [handles["state"]]}})

        stats = store.prune_outputs(str(tmp_path), cutoff_ns=time.time_ns() - 60 * 1_000_000_000)

        assert stats["sessions"] == 1 and stats["outputs"] == 6
        assert stats["deleted"] == 1 and stats["retained_referenced"] == 4 and stats["retained_recent"] == 1
        with pytest.raises(KeyError):
            store.get_output(session, handles["unreferenced"])
        for name, handle in handles.items():
            if name != "unreferenced":
                assert store.get_output(session, handle), name
        assert store.get_output(foreign, foreign_handle) == "other project"  # other project untouched


def test_prune_keeps_every_handle_named_by_a_compaction_range(tmp_path: Path):
    with SqliteSessionStore(tmp_path / "sessions.sqlite3") as store:
        session = store.create(str(tmp_path))
        handles = _old_outputs(store, session)
        store.append_message(
            session,
            ChatMessage(
                role="user",
                content=f"[compaction summary #1: ...] Full outputs: show_output(handle) for "
                f"#{handles['stub']}-#{handles['summary']}",
            ),
        )
        stats = store.prune_outputs(str(tmp_path), cutoff_ns=time.time_ns() - 60 * 1_000_000_000)

        # stub..summary survive through the range; state, unreferenced and recent are counted below
        assert stats["outputs"] == 6
        assert stats["deleted"] == 2  # 'state' (no reference here) + 'unreferenced'
        assert stats["retained_referenced"] == 3 and stats["retained_recent"] == 1
        for name in ("stub", "truncated", "summary", "recent"):
            assert store.get_output(session, handles[name]), name


def test_outputs_report_shows_the_retention_picture(store: SqliteSessionStore):
    session = store.create(str(Path.cwd()))
    handles = _old_outputs(store, session)
    store.append_message(
        session,
        ChatMessage(role="tool", content=f"[ok] see show_output(handle={handles['state']})", tool_call_id="c", name="bash"),
    )
    report = store.outputs_report(session)
    assert report == {"total": 6, "referenced": 1, "unreferenced": 5}
    assert store.state_keys(session) == []
