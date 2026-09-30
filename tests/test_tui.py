from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ironclad.config import NedsterConfig
from nedster_agent.tui import NedsterApp


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


@pytest.mark.asyncio
async def test_tui_keeps_events_in_details_and_displays_the_real_answer(tmp_path):
    config = NedsterConfig()
    config.runtime.project_root = tmp_path
    fleet = Mock()
    fleet.for_role.return_value = SimpleNamespace(name="nedster-worker")
    runtime = Mock(fleet=fleet)
    runtime.initialize = AsyncMock(return_value=fleet)
    runtime.respond = AsyncMock(return_value={"answer": "A grounded, useful answer.", "passed": True})

    with patch("nedster_agent.tui.NedsterRuntime", return_value=runtime):
        app = NedsterApp(config)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._event({"kind": "objective.received", "actor": "user"})
            await app._execute("Explain this")

            work_text = "\n".join(str(line) for line in app.query_one("#work").lines)
            assert "objective.received" not in work_text
            assert "A grounded, useful answer." in work_text

            app.action_details()
            await pilot.pause()
            detail_text = "\n".join(str(line) for line in app.query_one("#details").lines)
            assert "objective.received" in detail_text
            assert app.query_one("#details").display
            assert not app.query_one("#work").display


@pytest.mark.asyncio
async def test_tui_streams_plan_steps_tools_and_verification_into_conversation(tmp_path):
    config = NedsterConfig()
    config.runtime.project_root = tmp_path
    fleet = Mock()
    fleet.for_role.return_value = SimpleNamespace(name="nedster-worker")
    runtime = Mock(fleet=fleet)
    runtime.initialize = AsyncMock(return_value=fleet)

    with patch("nedster_agent.tui.NedsterRuntime", return_value=runtime):
        app = NedsterApp(config)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._event(
                {
                    "kind": "plan.created",
                    "actor": "router",
                    "plan": {
                        "steps": [
                            {"id": "inspect", "title": "Inspect the project"},
                            {"id": "test", "title": "Run focused tests"},
                        ]
                    },
                }
            )
            app._event(
                {
                    "kind": "step.started",
                    "actor": "reasoner",
                    "step": {"id": "inspect", "title": "Inspect the project", "role": "reasoner"},
                }
            )
            app._event({"kind": "tool.authorized", "actor": "policy", "tool": "git_diff"})
            app._event(
                {
                    "kind": "tool.completed",
                    "actor": "git_diff",
                    "result": {"tracked_count": 3, "untracked_count": 2},
                }
            )
            app._event(
                {
                    "kind": "step.finished",
                    "actor": "critic",
                    "step": {"id": "inspect", "title": "Inspect the project"},
                    "verdict": {"passed": True},
                }
            )
            await pilot.pause()

            work_text = "\n".join(line.text for line in app.query_one("#work").lines)
            assert "Plan" in work_text
            assert "Inspect the project" in work_text
            assert "Git Diff" in work_text
            assert "3 tracked, 2 untracked" in work_text
            assert "verified" in work_text


@pytest.mark.asyncio
async def test_tui_marks_grounded_final_answer_as_verified(tmp_path):
    config = NedsterConfig()
    config.runtime.project_root = tmp_path
    fleet = Mock()
    fleet.for_role.return_value = SimpleNamespace(name="nedster-worker")
    runtime = Mock(fleet=fleet)
    runtime.initialize = AsyncMock(return_value=fleet)
    runtime.respond = AsyncMock(
        return_value={
            "answer": "## Result\n\nThe inspected value is `42`.",
            "passed": True,
            "grounded": True,
            "completed_steps": 1,
            "total_steps": 1,
        }
    )

    with patch("nedster_agent.tui.NedsterRuntime", return_value=runtime):
        app = NedsterApp(config)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await app._execute("Inspect the value")
            await pilot.pause()

            work_text = "\n".join(line.text for line in app.query_one("#work").lines)
            assert "Result" in work_text
            assert "The inspected value is 42." in work_text
            assert "verified · 1/1 steps" in work_text
