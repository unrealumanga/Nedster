"""Final-review blockers B2/B3 on the REAL native adapter + SqliteSessionStore (scripted transport).

B2: the selected, locked, submitted, transcript and checkpoint session ids must be one id across
``/resume``, ``/new`` and a lazy first launch. B3: a turn is never admitted without the lock.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from ironclad.config import NedsterConfig
from ironclad.contracts import ChatMessage
from ironclad.session_store.sqlite_store import SqliteSessionStore, make_owner
from nedster_agent.tui import NedsterApp, native_session
from tests.engine.fakes import ScriptedTransport, Turn, call
from tests.ui.helpers import ready, wait_for


def _config(tmp_path: Path) -> NedsterConfig:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    config = NedsterConfig()
    config.runtime.project_root = project
    config.runtime.state_dir = tmp_path / "state"
    config.runtime.context_tokens = 8192  # no Ollama lookup
    config.runtime.soul = "off"
    return config


def _store(config: NedsterConfig) -> SqliteSessionStore:
    state = Path(config.runtime.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    return SqliteSessionStore(state / "sessions.sqlite3")


def _root(config: NedsterConfig) -> str:
    return str(Path(config.runtime.project_root).resolve())


def _seed(config: NedsterConfig, *conversations: str) -> list[str]:
    store = _store(config)
    try:
        ids = []
        for text in conversations:
            sid = store.create(_root(config), text)
            store.append_message(sid, ChatMessage(role="user", content=text))
            store.append_message(sid, ChatMessage(role="assistant", content=f"reply to {text}"))
            ids.append(sid)
        return ids
    finally:
        store.close()


def _contents(config: NedsterConfig, session_id: str) -> list[str]:
    store = _store(config)
    try:
        return [m.content for m in store.messages(session_id)]
    finally:
        store.close()


def _dump(config: NedsterConfig) -> dict[str, list[tuple[Any, ...]]]:
    """Every persisted row except the activity locks themselves."""
    connection = sqlite3.connect(Path(config.runtime.state_dir) / "sessions.sqlite3")
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


class _Factory:
    """Builds the real TUI native session and swaps in a scripted transport."""

    def __init__(self, script: list[Any], **kwargs: Any) -> None:
        self.transport = ScriptedTransport(script)
        self.kwargs = kwargs
        self.session: Any = None

    def __call__(self, config: NedsterConfig, emit: Any) -> Any:
        self.session = native_session(config, emit, **self.kwargs)
        self.session.core.engine.transport = self.transport
        return self.session


def _app(config: NedsterConfig, factory: _Factory) -> NedsterApp:
    return NedsterApp(config, session_factory=factory, files=lambda: [])


async def _turn(pilot: Any, app: NedsterApp, text: str) -> None:
    app.submit_text(text)
    await wait_for(pilot, lambda: not app.busy and app.status_bar.state == "idle")


def _ids_agree(app: NedsterApp, expected: str) -> None:
    session = app.session
    assert session.session_id == expected
    assert session.core.session_id == expected
    assert app._locked is not None and app._locked[1] == expected
    assert session.info()["session_id"] == expected


@pytest.mark.asyncio
async def test_resume_then_new_route_turns_to_the_selected_session(tmp_path):
    config = _config(tmp_path)
    a, b = _seed(config, "alpha", "bravo")
    target = Path(config.runtime.project_root) / "rng.py"
    target.write_bytes(b"x = 1\n")
    sha = hashlib.sha256(b"x = 1\n").hexdigest()[:16]
    factory = _Factory(
        [
            Turn(calls=[call("edit_file", "e1", path="rng.py", old="x = 1", new="x = 2", expected_sha=sha)]),
            Turn("resumed reply"),
            Turn("new reply"),
        ],
        session_id=a,
        lazy=False,
    )
    app = _app(config, factory)
    async with app.run_test(size=(120, 40)) as pilot:
        await ready(pilot, app)
        _ids_agree(app, a)

        app._command(f"/resume {b}")
        await pilot.pause()
        _ids_agree(app, b)
        assert app.session.store.lock_info(a) is None  # the old conversation's lock was released
        await _turn(pilot, app, "resume-marker")
        _ids_agree(app, b)

        app._command("/new")
        await pilot.pause()
        c = app.session.session_id
        assert c not in {a, b}
        _ids_agree(app, c)
        assert app.session.store.lock_info(b) is None
        await _turn(pilot, app, "new-marker")
        _ids_agree(app, c)

    assert _contents(config, a) == ["alpha", "reply to alpha"]  # A never touched
    b_rows = _contents(config, b)
    assert b_rows[:2] == ["bravo", "reply to bravo"] and "resume-marker" in b_rows and "resumed reply" in b_rows
    assert "new-marker" not in b_rows
    c_rows = _contents(config, c)
    assert "new-marker" in c_rows and "new reply" in c_rows and "resume-marker" not in c_rows

    store = _store(config)
    try:
        transcript_b = " ".join(str(entry.get("text", entry)) for entry in store.transcript(b))
        assert "resume-marker" in transcript_b
        assert "resume-marker" not in " ".join(str(e) for e in store.transcript(a))
    finally:
        store.close()
    connection = sqlite3.connect(Path(config.runtime.state_dir) / "sessions.sqlite3")
    try:
        checkpoint_sessions = {row[0] for row in connection.execute("SELECT session_id FROM ckpt_turns")}
    finally:
        connection.close()
    assert checkpoint_sessions == {b}  # the edit's checkpoint (and so /undo) belongs to B


@pytest.mark.asyncio
async def test_lazy_launch_then_pick_a_saved_session_creates_nothing_new(tmp_path):
    config = _config(tmp_path)
    (b,) = _seed(config, "bravo")
    factory = _Factory([Turn("picked reply")])  # lazy launch: no session row yet
    app = _app(config, factory)
    async with app.run_test(size=(120, 40)) as pilot:
        await ready(pilot, app)
        assert app.session.session_id is None and app._locked is None
        app._picker_chosen(b)
        await pilot.pause()
        _ids_agree(app, b)
        await _turn(pilot, app, "picked-marker")
        _ids_agree(app, b)

    store = _store(config)
    try:
        assert [s["id"] for s in store.list_sessions(_root(config))] == [b]
    finally:
        store.close()
    assert "picked-marker" in _contents(config, b)


@pytest.mark.asyncio
async def test_lazy_first_turn_fills_the_absent_id_only(tmp_path):
    config = _config(tmp_path)
    factory = _Factory([Turn("one"), Turn("two")])
    app = _app(config, factory)
    async with app.run_test(size=(120, 40)) as pilot:
        await ready(pilot, app)
        await _turn(pilot, app, "first")
        sid = app.session.session_id
        assert sid
        _ids_agree(app, sid)
        await _turn(pilot, app, "second")
        _ids_agree(app, sid)
    store = _store(config)
    try:
        assert [s["id"] for s in store.list_sessions(_root(config))] == [sid]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_session_locked_by_another_owner_refuses_the_turn_and_persists_nothing(tmp_path):
    config = _config(tmp_path)
    (a,) = _seed(config, "alpha")
    other = make_owner("tui")
    store = _store(config)
    assert store.acquire_lock(a, other)
    store.close()
    factory = _Factory([Turn("must not run")], session_id=a, lazy=False)
    app = _app(config, factory)
    async with app.run_test(size=(120, 40)) as pilot:
        await ready(pilot, app)
        before = _dump(config)
        assert app._locked is None
        app.submit_text("locked-marker")
        await pilot.pause()
        await pilot.pause()
        assert not app.busy
        assert app.query_one("#prompt").text == "locked-marker"  # the draft is kept
        assert factory.transport.requests == []  # the model was never called
        assert _dump(config) == before
        assert app.session.store.lock_info(a)["owner"] == other  # the other owner's lock is intact

    assert _contents(config, a) == ["alpha", "reply to alpha"]


@pytest.mark.asyncio
async def test_resume_into_a_locked_session_keeps_the_current_one_and_its_lock(tmp_path):
    config = _config(tmp_path)
    a, b = _seed(config, "alpha", "bravo")
    store = _store(config)
    assert store.acquire_lock(b, make_owner("tui"))
    store.close()
    factory = _Factory([Turn("a reply")], session_id=a, lazy=False)
    app = _app(config, factory)
    async with app.run_test(size=(120, 40)) as pilot:
        await ready(pilot, app)
        app._command(f"/resume {b}")
        await pilot.pause()
        _ids_agree(app, a)
        await _turn(pilot, app, "still-a")
        _ids_agree(app, a)
    assert "still-a" in _contents(config, a)
    assert "still-a" not in _contents(config, b)
