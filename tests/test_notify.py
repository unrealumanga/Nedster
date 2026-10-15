"""Completion notice: one message helper and a pure decision for off / bell / osc9."""

from __future__ import annotations

from nedster_agent.notify import BELL, completion_message, notification


def test_completion_message_is_plain_and_rounded():
    assert completion_message("completed", 42.4) == "Nedster: completed in 42s"
    assert completion_message("failed", 3.0) == "Nedster: failed in 3s"


def test_bell_mode_rings_only_after_the_threshold():
    assert notification("bell", "completed", 15.0, threshold=15.0, interactive=True) is None  # not longer
    assert notification("bell", "completed", 15.1, threshold=15.0, interactive=True) == BELL


def test_osc9_mode_embeds_the_message():
    data = notification("osc9", "failed", 90.0, threshold=15.0, interactive=True)
    assert data == "\x1b]9;Nedster: failed in 90s\x07"


def test_off_and_unknown_modes_and_machine_output_stay_silent():
    assert notification("off", "completed", 999.0, threshold=15.0, interactive=True) is None
    assert notification("", "completed", 999.0, threshold=15.0, interactive=True) is None
    assert notification("bogus", "completed", 999.0, threshold=15.0, interactive=True) is None
    # interactive=False is the machine-output guard: never emit escapes into --json/--jsonl/CLI pipes
    assert notification("bell", "completed", 999.0, threshold=15.0, interactive=False) is None
    assert notification("osc9", "completed", 999.0, threshold=15.0, interactive=False) is None
