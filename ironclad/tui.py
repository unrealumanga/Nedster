"""Backward-compatible TUI imports."""

from nedster_agent.tui import NedsterApp, run_tui

NedsterIroncladApp = NedsterApp

__all__ = ["NedsterApp", "NedsterIroncladApp", "run_tui"]
