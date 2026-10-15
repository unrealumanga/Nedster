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


# --- re-review B3: an expired lock taken over by another owner -----------------------------------


@pytest.mark.asyncio
async def test_expired_lock_taken_over_refuses_the_next_turn_and_persists_nothing(tmp_path):
    import time

    config = _config(tmp_path)
    (a,) = _seed(config, "alpha")
    factory = _Factory([Turn("must not run")], session_id=a, lazy=False)
    app = _app(config, factory)
    async with app.run_test(size=(120, 40)) as pilot:
        await ready(pilot, app)
        _ids_agree(app, a)
        connection = sqlite3.connect(Path(config.runtime.state_dir) / "sessions.sqlite3")
        try:  # our heartbeat is 61 s old: the store now treats the lock as expired
            connection.execute(
                "UPDATE session_locks SET heartbeat_ns = ? WHERE session_id = ?",
                (time.time_ns() - 61 * 1_000_000_000, a),
            )
            connection.commit()
        finally:
            connection.close()
        other_store = _store(config)
        other = make_owner("tui")
        try:
            assert other_store.acquire_lock(a, other)  # a second real owner takes the expired lock
        finally:
            other_store.close()
        before = _dump(config)

        app.submit_text("takeover-marker")  # before our next heartbeat
        await pilot.pause()
        await pilot.pause()

        assert not app.busy
        assert app._locked is None
        assert app.query_one("#prompt").text == "takeover-marker"
        assert "taken over by another Nedster" in app.transcript.plain_text()
        assert factory.transport.requests == []
        assert _dump(config) == before
        assert app.session.store.lock_info(a)["owner"] == other
    assert _contents(config, a) == ["alpha", "reply to alpha"]


# --- re-review S2: in-app resume restores the destination session's saved role ------------------


def _role_of(config: NedsterConfig, session_id: str) -> Any:
    store = _store(config)
    try:
        return store.get_state(session_id, "role")
    finally:
        store.close()


@pytest.mark.asyncio
async def test_resume_plan_session_from_code_session_restores_plan_and_its_gate(tmp_path):
    config = _config(tmp_path)
    a, p = _seed(config, "alpha", "planning")
    store = _store(config)
    store.set_state(a, "role", "code")
    store.set_state(p, "role", "plan")
    store.close()
    target = Path(config.runtime.project_root) / "rng.py"
    target.write_bytes(b"x = 1\n")
    sha = hashlib.sha256(b"x = 1\n").hexdigest()[:16]
    factory = _Factory(
        [
            Turn(calls=[call("edit_file", "e1", path="rng.py", old="x = 1", new="x = 2", expected_sha=sha)]),
            Turn("Here is the plan."),
            Turn("extra"),
            Turn("extra"),
        ],
        session_id=a,
        lazy=False,
    )
    app = _app(config, factory)
    async with app.run_test(size=(120, 40)) as pilot:
        await ready(pilot, app)
        assert app.session.info()["role"] == "code"

        app._command(f"/resume {p}")
        await pilot.pause()
        _ids_agree(app, p)
        assert app.session.info()["role"] == "plan"  # restored before any submission
        await _turn(pilot, app, "change x to 2")
        assert target.read_bytes() == b"x = 1\n"  # the write was denied by the plan gate
        assert _role_of(config, p) == "plan"  # and the saved role stayed plan

        app._command("/new")
        await pilot.pause()
        assert app.session.info()["role"] == "code"  # a new session gets the configured default
        app._command(f"/resume {a}")
        await pilot.pause()
        assert app.session.info()["role"] == "code"

        app.action_toggle_role()  # idle role changes persist immediately
        await pilot.pause()
        assert app.session.info()["role"] == "plan"
        assert _role_of(config, a) == "plan"
    assert _role_of(config, p) == "plan"


def test_explicit_role_override_wins_over_the_saved_role_on_switch(tmp_path):
    from ironclad.native import NativeSession as CoreSession

    config = _config(tmp_path)
    a, p = _seed(config, "alpha", "planning")
    store = _store(config)
    store.set_state(p, "role", "plan")
    store.close()
    core = CoreSession(config, session_id=a, role="code")
    try:
        core.switch_session(p)
        assert core.engine.role_name == "code"
    finally:
        import asyncio

        asyncio.run(core.aclose())


# --- re-review 2 B3: losing the lock while a turn is running --------------------------------------


class _GatedTransport(ScriptedTransport):
    """The first model call waits on ``gate`` (the UI is mid-turn, awaiting model output)."""

    def __init__(self, script: list[Any]) -> None:
        super().__init__(script)
        import asyncio

        self.gate = asyncio.Event()
        self.waiting = False

    async def chat(self, **kwargs: Any):  # type: ignore[override]
        if len(self.requests) == 0:
            self.waiting = True
            await self.gate.wait()
        async for event in super().chat(**kwargs):
            yield event


def _expire_and_take_over(config: NedsterConfig, session_id: str) -> str:
    import time

    connection = sqlite3.connect(Path(config.runtime.state_dir) / "sessions.sqlite3")
    try:
        connection.execute(
            "UPDATE session_locks SET heartbeat_ns = ? WHERE session_id = ?",
            (time.time_ns() - 61 * 1_000_000_000, session_id),
        )
        connection.commit()
    finally:
        connection.close()
    other_store = _store(config)
    other = make_owner("tui")
    try:
        assert other_store.acquire_lock(session_id, other)
    finally:
        other_store.close()
    return other


@pytest.mark.asyncio
@pytest.mark.parametrize("ui_cancels", [True, False], ids=["heartbeat-cancels", "engine-guard-only"])
async def test_lock_lost_mid_turn_stops_the_turn_and_writes_nothing_more(tmp_path, monkeypatch, ui_cancels):
    config = _config(tmp_path)
    (a,) = _seed(config, "alpha")
    after = Path(config.runtime.project_root) / "after-loss.txt"
    factory = _Factory([], session_id=a, lazy=False)
    gated = _GatedTransport([
        Turn(calls=[call("write_file", "w1", path="after-loss.txt", content="written after the loss\n")]),
        Turn("done"),
    ])
    factory.transport = gated
    app = _app(config, factory)
    if not ui_cancels:  # prove the engine-side guard alone blocks the write
        monkeypatch.setattr(NedsterApp, "cancel_turn", lambda self: None)
    async with app.run_test(size=(120, 40)) as pilot:
        await ready(pilot, app)
        app.submit_text("write after-loss.txt")
        await wait_for(pilot, lambda: gated.waiting)  # admitted; awaiting scripted model output
        persisted = _contents(config, a)
        assert "write after-loss.txt" in persisted  # the admitted turn's user message

        other = _expire_and_take_over(config, a)
        before = _dump(config)
        app._last_lock_refresh = 0.0  # the UI heartbeat runs now and sees the loss
        app._heartbeat_lock()
        assert app._locked is None
        gated.gate.set()
        await wait_for(pilot, lambda: not app.busy and app.status_bar.state == "idle")

        assert not after.exists()
        assert _contents(config, a) == persisted  # nothing persisted after the loss
        assert _dump(config) == before
        assert app._turn_outcome == "cancelled"
        assert app.session.store.lock_info(a)["owner"] == other  # the foreign lock is intact
        if ui_cancels:
            assert "the running turn was stopped" in app.transcript.plain_text()


@pytest.mark.asyncio
async def test_engine_guard_blocks_persisting_and_dispatch_once_ownership_is_lost(tmp_path):
    """Core level: the guard flips while the model 'thinks'; its tool call is never dispatched."""
    from ironclad.native import NativeSession as CoreSession

    config = _config(tmp_path)
    (a,) = _seed(config, "alpha")
    owned = {"ok": True}

    def lose(_messages: Any) -> Turn:
        owned["ok"] = False
        return Turn(calls=[call("write_file", "w1", path="after-loss.txt", content="x\n")])

    core = CoreSession(config, session_id=a)
    try:
        core.engine.transport = ScriptedTransport([lose, Turn("done")])
        core.set_owner_guard(lambda sid: owned["ok"] and sid == a)
        result = await core.run("write it")
        assert result.outcome.value == "cancelled"
        assert "session lock lost" in " ".join(result.reasons)
        assert not (Path(config.runtime.project_root) / "after-loss.txt").exists()
        assert [m.content for m in core.store.messages(a)] == ["alpha", "reply to alpha", "write it"]
    finally:
        await core.aclose()
