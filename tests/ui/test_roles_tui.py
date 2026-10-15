"""Tests for role presets in the TUI: /role, /act, Shift+Tab toggle."""

from __future__ import annotations

from pathlib import Path

from textual.widgets import Static

from tests.ui.helpers import Harness, ready, settle, transcript


def _message(app) -> str:
    return str(app.query_one("#message", Static).render())


# --------------------------------------------------------------------------- /role


async def test_role_command_lists_roles_and_marks_active(tmp_path: Path, monkeypatch):
    """No-argument /role lists all roles with * on the active one."""
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app._command("/role")
        await settle(pilot)
        text = transcript(h.app)
        for role_name in ("code", "plan", "review", "explain"):
            assert f"/role {role_name}" in text or f"role {role_name}" in text.lower()


async def test_role_switches_to_plan_and_status_shows_plan(tmp_path: Path, monkeypatch):
    """Switching to plan via /role makes the status show PLAN."""
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app._command("/role plan")
        await settle(pilot)
        text = transcript(h.app)
        assert "Mode: PLAN" in text or "PLAN" in text


async def test_role_unknown_reports_valid_names(tmp_path: Path, monkeypatch):
    """An unknown role name shows an error listing the valid names."""
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app._command("/role zebra")
        await settle(pilot)
        text = transcript(h.app)
        assert "unknown" in text.lower() or "valid" in text.lower()


# --------------------------------------------------------------------------- busy refusal


async def test_role_busy_refuses_during_turn(tmp_path: Path, monkeypatch):
    """While the engine is running, /role refuses with a message."""
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        # Kick off a turn and hold it open
        h.app.submit_text("work")
        await settle(pilot)
        assert h.app.busy is True
        # Try to switch role while busy
        h.app._command("/role plan")
        await settle(pilot)
        text = transcript(h.app)
        assert "finish or stop" in text.lower() or "turn" in text.lower()


async def test_toggle_busy_refuses_during_turn(tmp_path: Path, monkeypatch):
    """Shift+Tab refuses while the engine is running."""
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("work")
        await settle(pilot)
        assert h.app.busy is True
        await pilot.press("shift", "tab")
        await settle(pilot)
        text = transcript(h.app)
        assert "finish or stop" in text.lower() or "turn" in text.lower()


# --------------------------------------------------------------------------- shift+tab toggle


async def test_toggle_plan_code(tmp_path: Path, monkeypatch):
    """Shift+Tab toggles between plan and code."""
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        # First toggle -> plan (call the action directly to avoid binding issues in headless)
        h.app.action_toggle_role()
        await settle(pilot)
        text = transcript(h.app)
        assert "Mode: PLAN" in text or "PLAN" in text

        # Second toggle -> code
        h.app.action_toggle_role()
        await settle(pilot)
        text = transcript(h.app)
        assert "Mode: CODE" in text or "CODE" in text


# --------------------------------------------------------------------------- /act


async def test_act_switches_to_code_and_submits(tmp_path: Path, monkeypatch):
    """/act switches to code mode and submits 'Execute the plan above.'"""
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        # First switch to plan role
        h.app._command("/role plan")
        await settle(pilot)

        # Then /act should switch back to code and submit
        h.app._command("/act")
        await settle(pilot)

        # The fake session should have recorded the submission
        assert len(h.fake.submitted) >= 1
        assert "Execute the plan above." in h.fake.submitted[-1]


async def test_act_busy_refuses(tmp_path: Path, monkeypatch):
    """/act refuses while the engine is running."""
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("work")
        await settle(pilot)
        assert h.app.busy is True
        h.app._command("/act")
        await settle(pilot)
        text = transcript(h.app)
        assert "finish or stop" in text.lower() or "turn" in text.lower()


# --------------------------------------------------------------------------- help


async def test_help_lists_role_and_act(tmp_path: Path, monkeypatch):
    """/help lists /role and /act commands."""
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app._command("/help")
        await settle(pilot)
        text = transcript(h.app)
        assert "/role" in text
        assert "/act" in text
        assert "Shift+Tab" in text
