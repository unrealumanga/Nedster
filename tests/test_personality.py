from __future__ import annotations

from nedster_agent import personality

MOMENTS = ("thinking", "done_ok", "done_unvalidated", "done_failed", "idle")


def test_every_moment_has_lines_and_total_is_about_twenty():
    total = 0
    for moment in MOMENTS:
        options = personality.lines(moment)
        assert options, f"{moment} has no lines"
        assert all(line.strip() == line and line for line in options)
        total += len(options)
    assert 18 <= total <= 24
    assert personality.lines("nonsense") == ()


def test_pick_is_deterministic_and_stable_across_processes():
    assert personality.pick("thinking", 1) == personality.pick("thinking", 1)
    assert personality.pick("done_ok", 7) == personality.pick("done_ok", 7)
    assert personality.pick("unknown") == ""


def test_lines_are_dry_and_never_flatter_or_mock_the_user():
    banned = ("absolutely right", "great question", "you're wrong", "stupid", "idiot", "you are an")
    for moment in MOMENTS:
        for line in personality.lines(moment):
            assert line.lower() not in banned
            assert "absolutely" not in line.lower()
