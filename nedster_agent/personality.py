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
    "done_ok": (
        "Tests passed. I'm as surprised as you.",
        "Done and verified. Enjoy it while it lasts.",
        "Green suite. I'll take the win.",
        "It works. Statistically speaking.",
        "All checks passed. Suspiciously smooth.",
    ),
    "done_unvalidated": (
        "Changed files, ran nothing. Trust me? Neither would I.",
        "Edits made, tests untouched. That's on the record.",
        "I changed things. The tests are a rumour at this point.",
        "Written, not verified. The suite is still asleep.",
    ),
    "done_failed": (
        "That didn't work. My confidence was misplaced.",
        "Failed. On the bright side, it failed honestly.",
        "No luck. The code won this round.",
        "Broken. Not the first time.",
    ),
    "idle": (
        "Idle. Nothing is on fire.",
        "Waiting. Patiently, mostly.",
        "Nothing to do. I'll be over here.",
        "Idle. No opinions pending.",
    ),
}


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
