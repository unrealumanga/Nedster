from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ironclad.config import NedsterConfig
from nedster_agent.tui import NedsterApp


@pytest.mark.asyncio
async def test_tui_keeps_identity_work_status_and_keys_at_80_columns(tmp_path):
    config = NedsterConfig()
    config.runtime.project_root = tmp_path
    fleet = Mock()
    fleet.for_role.return_value = SimpleNamespace(name="nedster-worker")
    runtime = Mock(fleet=fleet)
    runtime.initialize = AsyncMock(return_value=fleet)

    with patch("nedster_agent.tui.NedsterRuntime", return_value=runtime):
        app = NedsterApp(config)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert "Nedster" in str(app.query_one("#identity").render())
            assert app.query_one("#work").display
            assert app.query_one("#message").display
            assert app.query_one("Footer").display
