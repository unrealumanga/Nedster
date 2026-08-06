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
