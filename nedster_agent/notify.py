"""Completion notices for the terminal.

Pure helpers so the behaviour (off / bell / osc9, the threshold, and suppression in machine modes)
is testable without a terminal. The TUI writes the returned escape sequence to the console; the
CLI ``run --json``/``--jsonl`` paths never touch this module.
"""

from __future__ import annotations

BELL = "\a"
OSC9 = "\x1b]9;{message}\x07"
MODES = ("off", "bell", "osc9")


def completion_message(outcome: str, seconds: float) -> str:
    """The one-line completion notice. `outcome` is short: 'completed', 'failed', 'cancelled'."""
    return f"Nedster: {outcome} in {seconds:.0f}s"


def notification(
    mode: str,
    outcome: str,
    seconds: float,
    *,
    threshold: float,
    interactive: bool,
) -> str | None:
    """The bytes to write for a finished turn, or None when it should stay silent.

    Silent when the mode is off/unknown, when ``interactive`` is false (machine output), or when
    the turn did not take longer than ``threshold`` seconds.
    """
    mode = (mode or "off").strip().lower()
    if mode not in ("bell", "osc9") or not interactive or seconds <= threshold:
        return None
    message = completion_message(outcome, seconds)
    return OSC9.format(message=message) if mode == "osc9" else BELL
