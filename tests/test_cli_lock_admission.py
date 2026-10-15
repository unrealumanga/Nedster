"""Final-review blocker B3 (CLI): session admission fails closed, before any session write."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from ironclad.config import NedsterConfig
from ironclad.contracts import ChatMessage
from ironclad.native import NativeSession
from ironclad.session_store.sqlite_store import SqliteSessionStore, make_owner
from nedster_agent.cli import SessionActive, _lock_run_session, main

EXIT_RUNTIME = 1


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("NEDSTER_") or name == "OLLAMA_HOST":
            monkeypatch.delenv(name)
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("NEDSTER_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def _store(root: Path) -> SqliteSessionStore:
    (root / "data").mkdir(parents=True, exist_ok=True)
    return SqliteSessionStore(root / "data" / "sessions.sqlite3")


def _dump(root: Path) -> dict[str, list[tuple[Any, ...]]]:
    connection = sqlite3.connect(root / "data" / "sessions.sqlite3")
    try:
        tables = [
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            if row[0] not in {"session_locks", "sqlite_sequence"}
        ]
        dump = {t: sorted(connection.execute(f"SELECT * FROM {t}").fetchall(), key=repr) for t in tables}
        return {t: rows for t, rows in dump.items() if rows}  # an empty schema table is not a write
    finally:
        connection.close()


def _seed(root: Path, title: str = "busy") -> str:
    store = _store(root)
    try:
        sid = store.create(str(root.resolve()), title)
        store.append_message(sid, ChatMessage(role="user", content="in flight"))
        return sid
    finally:
        store.close()


def test_lock_helper_refuses_a_deleted_session_and_leaves_no_orphans(clean_env):
    sid = _seed(clean_env)
    store = _store(clean_env)
    try:
        assert store.delete_session(sid)
        before = _dump(clean_env)
        with pytest.raises(SessionActive, match="no longer exists"):
            _lock_run_session(store, sid)
        assert store.lock_info(sid) is None
    finally:
        store.close()
    assert _dump(clean_env) == before
    connection = sqlite3.connect(clean_env / "data" / "sessions.sqlite3")
    try:
        orphans = connection.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id NOT IN (SELECT id FROM sessions)"
        ).fetchone()[0]
    finally:
        connection.close()
    assert orphans == 0


def test_lock_helper_treats_lock_layer_errors_and_missing_ids_as_refusals():
    class Broken:
        def acquire_lock(self, session_id: str, owner: str) -> bool:
            raise RuntimeError("database is locked")

    with pytest.raises(SessionActive, match="Could not lock session"):
        _lock_run_session(Broken(), "0123456789abcdef")
    with pytest.raises(SessionActive):
        _lock_run_session(Broken(), "")
    assert _lock_run_session(object(), "0123456789abcdef") is None  # no lock layer at all


def test_refused_admission_happens_before_constructor_writes(clean_env):
    sid = _seed(clean_env)
    store = _store(clean_env)
    other = make_owner("tui")
    assert store.acquire_lock(sid, other)
    store.close()
    before = _dump(clean_env)
    config = NedsterConfig()
    config.runtime.project_root = clean_env
    config.runtime.state_dir = clean_env / "data"
    config.runtime.context_tokens = 8192

    with pytest.raises(SessionActive, match="active in another Nedster"):
        NativeSession(config, session_id=sid, title="renamed by a refused run", admit=_lock_run_session)

    assert _dump(clean_env) == before  # no title change, no instruction/soul snapshots, no rows
    store = _store(clean_env)
    try:
        assert store.lock_info(sid)["owner"] == other
    finally:
        store.close()


def test_run_refused_at_admission_exits_1_and_writes_nothing(clean_env, capsys, monkeypatch):
    """The lock is taken between main's pre-check and the run (the race the pre-check cannot close)."""
    sid = _seed(clean_env)
    store = _store(clean_env)
    assert store.acquire_lock(sid, make_owner("tui"))
    store.close()
    monkeypatch.setattr(SqliteSessionStore, "is_active", lambda self, session_id, **_: False)
    monkeypatch.setenv("NEDSTER_CONTEXT_TOKENS", "8192")
    before = _dump(clean_env)

    code = main(["run", "--session", sid[:8], "--name", "renamed", "--project", str(clean_env), "hi"])

    assert code == EXIT_RUNTIME
    assert "is active in another Nedster" in capsys.readouterr().err
    assert _dump(clean_env) == before


def test_run_heartbeat_verify_refuses_after_an_expired_lock_is_taken_over(clean_env):
    import time

    sid = _seed(clean_env)
    store = _store(clean_env)
    try:
        heartbeat = _lock_run_session(store, sid)
        assert heartbeat is not None
        heartbeat.verify()  # still ours
        connection = sqlite3.connect(clean_env / "data" / "sessions.sqlite3")
        try:
            connection.execute(
                "UPDATE session_locks SET heartbeat_ns = ? WHERE session_id = ?",
                (time.time_ns() - 61 * 1_000_000_000, sid),
            )
            connection.commit()
        finally:
            connection.close()
        other_store = _store(clean_env)
        try:
            assert other_store.acquire_lock(sid, make_owner("tui"))
        finally:
            other_store.close()
        with pytest.raises(SessionActive, match="taken over"):
            heartbeat.verify()
        heartbeat.close()
    finally:
        store.close()
