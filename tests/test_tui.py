import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ironclad.config import NedsterConfig
from nedster_agent.tui import NedsterApp
from nedster_agent.widgets.transcript import ToolRow


@pytest.fixture(autouse=True)
def _offline(tmp_path, monkeypatch):
    # Never probe real sandbox backends or write history into the user's home in tests.
    monkeypatch.setattr("nedster_agent.widgets.session._probe_sandbox", lambda config: ("docker", "full"))
    original = NedsterConfig.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.runtime.state_dir = tmp_path / "state"

    monkeypatch.setattr(NedsterConfig, "__init__", init)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40), (200, 60)])
async def test_tui_keeps_identity_work_status_and_keys_at_supported_sizes(tmp_path, size):
    config = NedsterConfig()
    config.runtime.project_root = tmp_path
    fleet = Mock()
    fleet.for_role.return_value = SimpleNamespace(name="nedster-worker")
    runtime = Mock(fleet=fleet)
    runtime.initialize = AsyncMock(return_value=fleet)
    runtime.respond = AsyncMock(return_value={"answer": "Hello!", "passed": True})

    with patch("nedster_agent.tui.NedsterRuntime", return_value=runtime):
        app = NedsterApp(config)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            assert "Nedster" in str(app.query_one("#identity").render())
            assert app.query_one("#work").display
            assert app.query_one("#message").display
            assert app.query_one("Footer").display
            assert app.query_one("#prompt").cursor_blink is False
            assert not app.query_one("#details").display


@pytest.mark.asyncio
async def test_tui_shows_only_size_remedy_below_minimum(tmp_path):
    config = NedsterConfig()
    config.runtime.project_root = tmp_path
    runtime = Mock()
    runtime.initialize = AsyncMock(side_effect=RuntimeError("not needed"))

    with patch("nedster_agent.tui.NedsterRuntime", return_value=runtime):
        app = NedsterApp(config, ascii_only=True, no_color=True)
        async with app.run_test(size=(79, 23)) as pilot:
            await pilot.pause()
            assert app.query_one("#too-small").display
            assert not app.query_one("#work").display
            assert not app.query_one("Footer").display


def test_tui_converts_activity_symbols_in_ascii_mode():
    app = NedsterApp(NedsterConfig(), ascii_only=True)

    rendered = app._text("● ◇ ◆ → ✓ ✕ ↻ › … · —")

    assert rendered == "* > > -> OK X ~ > ... | -"


def _legacy_runtime(respond_events, result):
    """A NedsterRuntime double whose respond() emits legacy dict events through on_event."""
    fleet = Mock()
    fleet.for_role.return_value = SimpleNamespace(name="nedster-worker")
    fleet.as_rows.return_value = [("coder", "nedster-worker", "ollama")]
    runtime = Mock(fleet=fleet)
    runtime.initialize = AsyncMock(return_value=fleet)

    def factory(config, on_event):
        async def respond(objective):
            for event in respond_events:
                on_event(event)
            return result

        runtime.respond = respond
        return runtime

    return factory


def _runtime_with(respond):
    fleet = Mock()
    fleet.for_role.return_value = SimpleNamespace(name="nedster-worker")
    runtime = Mock(fleet=fleet)
    runtime.initialize = AsyncMock(return_value=fleet)
    runtime.respond = respond
    return runtime


async def _settle(pilot, times=5):
    for _ in range(times):
        await pilot.pause()


async def _wait_for(pilot, predicate, timeout=5.0):
    import time

    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await pilot.pause(0.01)
    await pilot.pause()


async def _ready(pilot, app):
    await _wait_for(pilot, lambda: app.status_bar.state in {"idle", "setup needed"})


async def _idle(pilot, app):
    await _wait_for(pilot, lambda: not app.busy and not app.queue and app.status_bar.state == "idle")


def _transcript(app):
    return app.transcript.plain_text()


def _screen(app):
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


@pytest.mark.asyncio
async def test_tui_keeps_events_in_details_and_displays_the_real_answer(tmp_path):
    config = NedsterConfig()
    config.runtime.project_root = tmp_path
    factory = _legacy_runtime(
        [{"kind": "objective.received", "actor": "user"}],
        {"answer": "A grounded, useful answer.", "passed": True},
    )
    with patch("nedster_agent.tui.NedsterRuntime", side_effect=factory):
        app = NedsterApp(config)
        async with app.run_test(size=(120, 40)) as pilot:
            await _ready(pilot, app)
            app.submit_text("Explain this")
            await _idle(pilot, app)

            work_text = _transcript(app)
            assert "objective.received" not in work_text
            assert "A grounded, useful answer." in work_text
            assert "[ANSWERED]" in work_text

            app.action_details()
            await pilot.pause()
            detail_text = "\n".join(str(line) for line in app.query_one("#details").lines)
            assert "objective.received" in detail_text
            assert app.query_one("#details").display
            assert not app.query_one("#work").display


@pytest.mark.asyncio
async def test_tui_streams_plan_steps_tools_and_review_into_conversation(tmp_path):
    config = NedsterConfig()
    config.runtime.project_root = tmp_path
    events = [
        {"kind": "plan.created", "actor": "router", "plan": {"steps": [
            {"id": "inspect", "title": "Inspect the project"},
            {"id": "test", "title": "Run focused tests"},
        ]}},
        {"kind": "step.started", "actor": "reasoner",
         "step": {"id": "inspect", "title": "Inspect the project", "role": "reasoner"}},
        {"kind": "tool.authorized", "actor": "policy", "tool": "git_diff"},
        {"kind": "tool.completed", "actor": "git_diff", "result": {"tracked_count": 3, "untracked_count": 2}},
        {"kind": "step.finished", "actor": "critic", "step": {"id": "inspect", "title": "Inspect the project"},
         "verdict": {"passed": True}},
    ]
    factory = _legacy_runtime(events, {"answer": "done", "passed": True, "completed_steps": 1, "total_steps": 2})
    with patch("nedster_agent.tui.NedsterRuntime", side_effect=factory):
        app = NedsterApp(config)
        async with app.run_test(size=(120, 40)) as pilot:
            await _ready(pilot, app)
            app.submit_text("Inspect")
            await _idle(pilot, app)

            work_text = _transcript(app)
            assert "Plan" in work_text
            assert "Inspect the project" in work_text
            assert "git_diff" in work_text
            assert "3 tracked, 2 untracked" in work_text
            # A critic pass is a review, not validation: the UI never claims "verified" for it.
            assert "reviewed" in work_text and "verified" not in work_text
            row = app.query_one(ToolRow)
            assert row.status == "ok" and "✓" in row.summary_line().plain


@pytest.mark.asyncio
async def test_tui_marks_grounded_final_answer_as_completed(tmp_path):
    config = NedsterConfig()
    config.runtime.project_root = tmp_path
    factory = _legacy_runtime([], {
        "answer": "## Result\n\nThe inspected value is `42`.",
        "passed": True,
        "grounded": True,
        "completed_steps": 1,
        "total_steps": 1,
    })
    with patch("nedster_agent.tui.NedsterRuntime", side_effect=factory):
        app = NedsterApp(config)
        async with app.run_test(size=(120, 40)) as pilot:
            await _ready(pilot, app)
            app.submit_text("Inspect the value")
            await _idle(pilot, app)

            screen = _screen(app)
            assert "Result" in screen
            assert "The inspected value is 42." in screen
            assert "[COMPLETED]" in screen and "1/1 steps" in screen


@pytest.mark.asyncio
async def test_legacy_failed_command_renders_as_failure_not_success(tmp_path):
    """B15: tool.completed with returncode=1 / tests_run=0 must never show a checkmark."""
    config = NedsterConfig()
    config.runtime.project_root = tmp_path
    events = [
        {"kind": "tool.authorized", "tool": "run_tests"},
        {"kind": "tool.completed", "actor": "run_tests", "result": {"returncode": 1, "tests_run": 0}},
        {"kind": "tool.authorized", "tool": "run_command"},
        {"kind": "tool.denied", "tool": "run_command", "reason": "not allowed"},
    ]
    factory = _legacy_runtime(events, {"answer": "tests failed", "passed": False})
    with patch("nedster_agent.tui.NedsterRuntime", side_effect=factory):
        app = NedsterApp(config)
        async with app.run_test(size=(80, 24)) as pilot:
            await _ready(pilot, app)
            app.submit_text("run the tests")
            await _idle(pilot, app)
            rows = list(app.query(ToolRow))
            assert [row.status for row in rows] == ["failed", "failed"]
            first = rows[0].summary_line().plain
            assert "✗ run_tests" in first and "exit 1" in first
            assert "denied" in rows[1].summary_line().plain
            assert "✓" not in _transcript(app)
            assert "[INCOMPLETE]" in _transcript(app)


@pytest.mark.asyncio
async def test_legacy_busy_submit_is_queued_not_cleared(tmp_path):
    """B15: submitting while busy used to clear the composer and drop the message."""
    config = NedsterConfig()
    config.runtime.project_root = tmp_path
    gate = asyncio.Event()
    seen: list[str] = []

    async def respond(objective):
        seen.append(objective)
        if len(seen) == 1:
            await gate.wait()
        return {"answer": f"answer to {objective}", "passed": True}

    with patch("nedster_agent.tui.NedsterRuntime", return_value=_runtime_with(respond)):
        app = NedsterApp(config)
        async with app.run_test(size=(80, 24)) as pilot:
            await _ready(pilot, app)
            app.submit_text("first")
            await _wait_for(pilot, lambda: seen == ["first"])
            app.submit_text("second")
            await _settle(pilot)
            assert seen == ["first"] and "queued: second" in _transcript(app)
            gate.set()
            await _idle(pilot, app)
            assert seen == ["first", "second"]
            assert "answer to second" in _transcript(app)


@pytest.mark.asyncio
async def test_legacy_escape_cancels_running_turn(tmp_path):
    config = NedsterConfig()
    config.runtime.project_root = tmp_path

    async def respond(objective):
        await asyncio.Event().wait()

    with patch("nedster_agent.tui.NedsterRuntime", return_value=_runtime_with(respond)):
        app = NedsterApp(config)
        async with app.run_test(size=(100, 30)) as pilot:
            await _ready(pilot, app)
            app.submit_text("forever")
            await _settle(pilot)
            await pilot.press("escape")
            await _idle(pilot, app)
            assert not app.busy and app.status_bar.state == "idle"
            assert "[CANCELLED]" in _transcript(app)
            assert app.is_running


@pytest.mark.asyncio
async def test_native_engine_is_wired_and_uses_the_session_store(tmp_path):
    from nedster_agent.tui import _WiredNativeSession

    config = NedsterConfig()
    config.runtime.project_root = tmp_path
    config.runtime.state_dir = tmp_path / "state"
    config.engine = "native"
    app = NedsterApp(config)
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        assert isinstance(app.session, _WiredNativeSession)
        assert app.session.session_id == app.session.core.session_id
        assert "SETUP NEEDED" not in str(app.query_one("#identity").render())
    assert (tmp_path / "state" / "sessions.sqlite3").exists()


# --- /resume replay, activity locks, interrupted-call repair (W2-RESUME) ---------------------

from pathlib import Path  # noqa: E402

from ironclad.contracts import ToolCall  # noqa: E402
from ironclad.session_store.sqlite_store import (  # noqa: E402
    INTERRUPTED_TEXT,
    SqliteSessionStore,
    make_owner,
)
from nedster_agent.widgets.transcript import RESUME_BANNER  # noqa: E402


class _FakeNativeSession:
    """Duck-typed NativeSession double on a real SqliteSessionStore: no model, no engine."""

    def __init__(self, config, _emit, *, session_id: str = "") -> None:
        self.model = "nedster-worker"
        state_dir = Path(config.runtime.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        self.store = SqliteSessionStore(state_dir / "sessions.sqlite3")
        self.project_root = str(Path(config.runtime.project_root).resolve())
        self.session_id = session_id or self.store.create(self.project_root)

    async def submit(self, text: str) -> None:  # pragma: no cover - /resume is idle-only
        raise AssertionError(f"replay must never re-run a turn (got {text!r})")

    async def cancel(self) -> None:
        return None

    async def close(self) -> None:
        self.store.close()

    def info(self):
        return {
            "model": self.model,
            "provider": "LOCAL",
            "sandbox": "docker",
            "isolation": "full",
            "context_used": 0,
            "context_max": 32_768,
            "session_id": self.session_id,
            "engine": "native",
        }

    def new(self) -> str:
        self.session_id = self.store.create(self.project_root)
        return self.session_id

    def list_sessions(self):
        return list(self.store.list_sessions(self.project_root))

    def resume(self, session_id=None) -> str:
        sessions = self.list_sessions()
        if not sessions:
            raise LookupError("no saved sessions for this project")
        if session_id is None:
            chosen = sessions[0]
        else:
            matches = [s for s in sessions if str(s["id"]).startswith(session_id)]
            if len(matches) != 1:
                raise LookupError(f"no unique session matches {session_id!r}")
            chosen = matches[0]
        self.session_id = str(chosen["id"])
        return self.session_id

    def set_model(self, name: str) -> None:
        self.model = name


def _resume_app(tmp_path):
    config = NedsterConfig()
    config.runtime.project_root = tmp_path
    config.runtime.state_dir = tmp_path / "state"
    config.engine = "native"
    return NedsterApp(config, session_factory=lambda cfg, emit: _FakeNativeSession(cfg, emit))


def _seed_session(app, *, title: str) -> str:
    """A stored session with a display transcript, created before this window opened."""
    session = app.session.store.create(str(Path(app.session.project_root)), title)
    app.session.store.append_transcript(session, "user", "inspect the widget")
    app.session.store.append_transcript(session, "assistant", "the widget is fine")
    app.session.store.append_transcript(session, "tool", "pytest -q", {"tool": "run_tests", "status": "ok"})
    return session


@pytest.mark.asyncio
async def test_resume_replays_the_stored_transcript_and_warns_it_is_stale(tmp_path):
    app = _resume_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await _ready(pilot, app)
        older = _seed_session(app, title="older work")
        app._command(f"/resume {older[:8]}")
        await _settle(pilot)

        text = _transcript(app)
        assert f"Resumed session {older[:8]}" in text
        assert "inspect the widget" in text and "the widget is fine" in text
        # a tool row is shown as history, never replayed with its stored output text
        assert "earlier tool result - not replayed" in text
        assert "pytest -q" not in text
        assert RESUME_BANNER in text
        # the replay is the stored transcript, not a re-execution
        assert "Turn failed" not in text
        assert app.session.session_id == older
        assert app.status_bar.state == "idle"


@pytest.mark.asyncio
async def test_resume_takes_the_activity_lock_and_delete_refuses(tmp_path):
    app = _resume_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await _ready(pilot, app)
        older = _seed_session(app, title="locked by this window")
        assert app.session.store.is_active(older) is False

        app._command(f"/resume {older[:8]}")
        await _settle(pilot)

        store = app.session.store
        assert store.is_active(older) is True
        assert store.lock_info(older)["pid"] == __import__("os").getpid()
        assert store.delete_session(older) is False  # refused: this window is using it


@pytest.mark.asyncio
async def test_resume_keeps_the_current_session_when_another_window_holds_the_lock(tmp_path):
    app = _resume_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await _ready(pilot, app)
        older = _seed_session(app, title="used elsewhere")
        foreign = app.session.store.acquire_lock(older, make_owner("tui"))
        assert foreign is True  # a live holder from another window (same host, live pid)

        current = app.session.session_id
        app._command(f"/resume {older[:8]}")
        await _settle(pilot)

        text = _transcript(app)
        assert "active in another Nedster window" in text
        assert app.session.session_id == current
        assert f"Resumed session {older[:8]}" not in text
        assert app.session.store.is_active(older) is True


@pytest.mark.asyncio
async def test_resume_unknown_id_warns_and_stays_put(tmp_path):
    app = _resume_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await _ready(pilot, app)
        _seed_session(app, title="the only one")
        current = app.session.session_id

        app._command("/resume deadbeef")
        await _settle(pilot)

        assert "no unique session matches" in _transcript(app)
        assert app.session.session_id == current


@pytest.mark.asyncio
async def test_resume_repairs_interrupted_tool_calls_before_replaying(tmp_path):
    app = _resume_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await _ready(pilot, app)
        store = app.session.store
        broken = store.create(app.session.project_root, "cut off mid-tool")
        store.append_transcript(broken, "user", "run the tests")
        from ironclad.contracts import ChatMessage

        store.append_message(broken, ChatMessage(role="user", content="run the tests"))
        store.append_message(
            broken,
            ChatMessage(role="assistant", content="", tool_calls=[ToolCall(id="c1", name="run_tests", arguments={})]),
        )

        app._command(f"/resume {broken[:8]}")
        await _settle(pilot)

        assert "Repaired 1 interrupted tool call(s)" in _transcript(app)
        repaired = store.messages(broken)
        assert [message.role for message in repaired] == ["user", "assistant", "tool"]
        assert repaired[2].content == INTERRUPTED_TEXT
        assert repaired[2].tool_call_id == "c1"
