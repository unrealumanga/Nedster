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
async def test_native_engine_without_wiring_shows_setup_remedy(tmp_path):
    config = NedsterConfig()
    config.runtime.project_root = tmp_path
    config.engine = "native"
    app = NedsterApp(config)
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        assert "SETUP NEEDED" in str(app.query_one("#identity").render())
        assert "native engine" in _transcript(app)
        app.submit_text("hello")
        await pilot.pause()
        assert app.query_one("#prompt").text == "hello"  # kept, not lost
