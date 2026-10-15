from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest

from ironclad.contracts import ChatMessage, ToolCall
from ironclad.session_store.sqlite_store import SCHEMA_VERSION, SqliteSessionStore, normalize_project_root


def _sample_messages() -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content="hello"),
        ChatMessage(
            role="assistant",
            content="running a tool",
            thinking="hmm",
            tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": "ls"})],
        ),
        ChatMessage(role="tool", content="file.py", tool_call_id="c1", name="bash"),
    ]


def test_message_round_trip_including_tool_calls(tmp_path: Path):
    with SqliteSessionStore(tmp_path / "sessions.sqlite3") as store:
        session = store.create(str(tmp_path))
        for message in _sample_messages():
            store.append_message(session, message)
        restored = store.messages(session)
    assert restored == _sample_messages()
    assert restored[1].tool_calls[0].arguments == {"command": "ls"}


def test_output_handles_increase_per_session(tmp_path: Path):
    with SqliteSessionStore(tmp_path / "sessions.sqlite3") as store:
        session = store.create(str(tmp_path))
        assert store.put_output(session, "bash", "c1", "first") == 1
        assert store.put_output(session, "bash", "c2", "second") == 2
        other = store.create(str(tmp_path))
        assert store.put_output(other, "bash", "c3", "other") == 1
        assert store.get_output(session, 1) == "first"
        assert store.get_output(session, 2) == "second"
        with pytest.raises(KeyError):  # review #16: a missing handle is an error, never ""
            store.get_output(session, 99)
        with pytest.raises(KeyError):  # handles are per session
            store.get_output(other, 2)


def test_list_sessions_filters_by_normalized_root(tmp_path: Path):
    root_a = tmp_path / "proj_a"
    root_b = tmp_path / "proj_b"
    root_a.mkdir()
    root_b.mkdir()
    with SqliteSessionStore(tmp_path / "sessions.sqlite3") as store:
        keep = store.create(str(root_a), title="keep")
        store.create(str(root_b), title="leak")
        listing = store.list_sessions(str(root_a) + "/")
        assert [entry["id"] for entry in listing] == [keep]
        assert listing[0]["title"] == "keep"
        assert listing[0]["project_root"] == normalize_project_root(str(root_a))
        assert store.latest_session(str(root_a)) == keep
        assert store.latest_session(str(tmp_path / "missing")) is None


def test_set_and_get_state_round_trip(tmp_path: Path):
    with SqliteSessionStore(tmp_path / "sessions.sqlite3") as store:
        session = store.create(str(tmp_path))
        store.set_state(session, "budget", {"turns": 3, "ok": True})
        assert store.get_state(session, "budget") == {"turns": 3, "ok": True}
        assert store.get_state(session, "absent", default="fallback") == "fallback"


def test_replace_messages_is_atomic_on_midway_failure(tmp_path: Path):
    with SqliteSessionStore(tmp_path / "sessions.sqlite3") as store:
        session = store.create(str(tmp_path))
        original = _sample_messages()
        for message in original:
            store.append_message(session, message)
        poisoned = [
            ChatMessage(role="user", content="fine"),
            ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(id="bad", name="t", arguments={"x": {1, 2}})],  # not JSON
            ),
        ]
        with pytest.raises(TypeError):
            store.replace_messages(session, poisoned)
        assert store.messages(session) == original
        store.replace_messages(session, [ChatMessage(role="user", content="compacted")])
        assert store.messages(session) == [ChatMessage(role="user", content="compacted")]


def test_data_persists_across_reopen(tmp_path: Path):
    path = tmp_path / "sessions.sqlite3"
    with SqliteSessionStore(path) as store:
        session = store.create(str(tmp_path), title="t")
        for message in _sample_messages():
            store.append_message(session, message)
        handle = store.put_output(session, "bash", "c1", "payload")
        store.set_state(session, "k", [1, 2])
    with SqliteSessionStore(path) as store:
        assert store.messages(session) == _sample_messages()
        assert store.get_output(session, handle) == "payload"
        assert store.get_state(session, "k") == [1, 2]
        assert store.latest_session(str(tmp_path)) == session


def test_fresh_file_migrates_to_current_schema(tmp_path: Path):
    path = tmp_path / "fresh.sqlite3"
    with SqliteSessionStore(path) as store:
        assert store.schema_version() == SCHEMA_VERSION
    with closing(sqlite3.connect(path)) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert {"schema_version", "sessions", "messages", "outputs", "state"} <= tables


def test_concurrent_appends_do_not_collide_on_seq(tmp_path: Path):
    with SqliteSessionStore(tmp_path / "sessions.sqlite3") as store:
        session = store.create(str(tmp_path))
        threads = [
            threading.Thread(
                target=lambda: [
                    store.append_message(session, ChatMessage(role="user", content="x"))
                    for _ in range(50)
                ]
            )
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        messages = store.messages(session)
    assert len(messages) == 200


def test_search_sessions_reports_counts_and_first_last_text(tmp_path: Path):
    with SqliteSessionStore(tmp_path / "sessions.sqlite3") as store:
        session = store.create(str(tmp_path), title="Fix the range")
        store.append_message(session, ChatMessage(role="user", content="  fix   the range  "))
        store.append_message(session, ChatMessage(role="assistant", content="done"))
        store.append_message(session, ChatMessage(role="user", content="and the boundary"))
        rows = store.search_sessions(str(tmp_path), "range")
        assert [row["id"] for row in rows] == [session]
        assert rows[0]["message_count"] == 3
        assert rows[0]["first_user_text"] == "  fix   the range  "
        assert rows[0]["last_user_text"] == "and the boundary"


def test_search_sessions_matches_id_prefix_title_and_first_message(tmp_path: Path):
    with SqliteSessionStore(tmp_path / "sessions.sqlite3") as store:
        titled = store.create(str(tmp_path), title="Write docs")
        store.append_message(titled, ChatMessage(role="user", content="anything"))
        blank = store.create(str(tmp_path))
        store.append_message(blank, ChatMessage(role="user", content="explain the widget"))
        assert {row["id"] for row in store.search_sessions(str(tmp_path), "write")} == {titled}
        assert {row["id"] for row in store.search_sessions(str(tmp_path), "widget")} == {blank}
        assert {row["id"] for row in store.search_sessions(str(tmp_path), titled[:6])} == {titled}
        assert {row["id"] for row in store.search_sessions(str(tmp_path), "")} == {titled, blank}


def test_search_sessions_excludes_empty_and_respects_the_limit(tmp_path: Path):
    with SqliteSessionStore(tmp_path / "sessions.sqlite3") as store:
        empty = store.create(str(tmp_path), title="no messages")
        full = store.create(str(tmp_path), title="has messages")
        store.append_message(full, ChatMessage(role="user", content="hi"))
        assert [row["id"] for row in store.search_sessions(str(tmp_path))] == [full]
        assert [row["id"] for row in store.search_sessions(str(tmp_path), exclude_empty=False)] == [full, empty]
        assert len(store.search_sessions(str(tmp_path), limit=1)) == 1


def test_latest_non_empty_skips_the_newest_empty_session(tmp_path: Path):
    with SqliteSessionStore(tmp_path / "sessions.sqlite3") as store:
        older = store.create(str(tmp_path))
        store.append_message(older, ChatMessage(role="user", content="older work"))
        store.create(str(tmp_path))  # newer, empty
        assert store.latest_session(str(tmp_path)) != older  # the raw latest is the empty one
        assert store.latest_non_empty(str(tmp_path)) == older
        assert store.latest_non_empty(str(tmp_path / "missing")) is None

