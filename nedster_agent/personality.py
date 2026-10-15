"""Opt-in personality lines for tier ``ui`` (zero model cost).

These are short, dry one-liners shown only in TUI status text and completion notices. They never
reach the model, the answer text, outcome labels, evidence, errors, file contents or commits. The
pick is a stable hash of the moment and turn number, so the same turn always shows the same line.
The tone stays dry and self-deprecating about Nedster or the code, never about the user, and never
flatters.
"""

from __future__ import annotations

import hashlib

LINES: dict[str, tuple[str, ...]] = {
    "thinking": (
        "Reading your code. Judging silently.",
        "Thinking. Slower than I look, which is saying something.",
        "Digging through the codebase. No promises.",
        "Working on it. The code has opinions.",
        "On it. Keeping the commentary to a minimum.",
    ),
    # Only "done_ok" may mention passing checks: it is shown solely for the test-backed "completed" outcome.
    # Every other moment is evidence-neutral; deterministic acceptance/evidence rendering alone describes
    # what was actually run.
    "done_ok": (
        "Tests passed. I'm as surprised as you.",
        "Done and verified. Enjoy it while it lasts.",
        "Green suite. I'll take the win.",
        "It works. Statistically speaking.",
        "All checks passed. Suspiciously smooth.",
    ),
    "done_answered": (
        "Answered. Citations are in the text, opinions are my own.",
        "That's my answer. Weigh it like you would anyone's.",
        "Done talking. The details are above.",
    ),
    "done_unvalidated": (
        "Changed things, verified nothing. Trust me? Neither would I.",
        "Edits made, not verified. That's on the record.",
        "I changed things. Verification is a rumour at this point.",
        "Written, not verified. Read it before you trust it.",
    ),
    "done_failed": (
        "Didn't land. My confidence was misplaced.",
        "Not done. On the bright side, it stopped honestly.",
        "No luck this round.",
        "Stopped short. Not the first time.",
    ),
    "idle": (
        "Idle. Nothing is on fire.",
        "Waiting. Patiently, mostly.",
        "Nothing to do. I'll be over here.",
        "Idle. No opinions pending.",
    ),
}


#: Turn outcome -> personality moment. Only the test-backed "completed" outcome maps to "done_ok".
OUTCOME_MOMENTS: dict[str, str] = {
    "completed": "done_ok",
    "answered": "done_answered",
    "no_change_needed": "done_answered",
    "unvalidated": "done_unvalidated",
    "incomplete": "done_failed",
    "failed": "done_failed",
    "cancelled": "done_failed",
}


def moment_for_outcome(outcome: str) -> str:
    """The personality moment for a finished turn's outcome ("" when none applies)."""
    return OUTCOME_MOMENTS.get(outcome, "")


def lines(moment: str) -> tuple[str, ...]:
    """All lines for ``moment`` (empty tuple for an unknown moment)."""
    return LINES.get(moment, ())


def pick(moment: str, turn: int = 0) -> str:
    """One deterministic line for ``moment``; stable across processes for the same turn."""
    options = lines(moment)
    if not options:
        return ""
    digest = hashlib.sha256(f"{moment}:{turn}".encode()).digest()
    return options[int.from_bytes(digest[:4], "big") % len(options)]
