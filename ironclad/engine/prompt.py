"""System prompt: fixed instructions plus project facts computed once per session.

The text must be byte-identical across turns of a session (stable prefix), so facts are computed once,
persisted in session state, and never contain timestamps or other volatile values.
"""

from __future__ import annotations

import platform
import unicodedata
from collections.abc import Mapping
from pathlib import Path

MAX_TOP_LEVEL = 40
MAX_NAME_CHARS = 64

INSTRUCTIONS = """\
You are Nedster, a local coding agent working inside one project directory through tools.

How to work:
- Use tools to inspect before you change anything. Read a file before editing it and pass the hash you read.
- Make the smallest change that satisfies the request. Do not invent files or results.
- After changing code, run the project's tests (run_tests) or a validating command (run_command with
  "validation": true). A change is only accepted when validation passes AFTER your latest change.
- If a tool result is [invalid] or [denied], read the diagnostic and correct the call; do not repeat the same call.
- Old tool outputs may be shown as "[output #N elided ...]". Call show_output(N) if you need them again.
- If the requested change is already in place, call declare_no_change only after passing tests prove it.
- When the request is a question, answer it from what the tools showed you. Say what you did not verify.
- Finish with a short plain answer and no tool calls. The engine appends the list of changed files and the
  validation result from its own records; do not claim tests passed unless a tool result showed it."""

_RUNNER_MARKERS = (
    ("pyproject.toml", "python (pytest)"),
    ("setup.py", "python (pytest)"),
    ("package.json", "node (npm test)"),
    ("Cargo.toml", "rust (cargo test)"),
    ("go.mod", "go (go test ./...)"),
)


def _clean(text: str) -> str:
    """Replace control/format/separator characters (newlines, U+2028, bidi overrides) so a value stays one line."""
    return "".join("?" if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") else ch for ch in text)


def safe_name(name: str) -> str:
    """A file name as it may appear in the system prompt: one line, bounded length."""
    name = _clean(name)
    return name if len(name) <= MAX_NAME_CHARS else name[: MAX_NAME_CHARS - 1] + "…"


def project_facts(root: str | Path | None) -> dict[str, str]:
    """Small, deterministic facts about the project. Never raises for unreadable directories."""
    facts = {"os": platform.system() or "unknown"}
    if root is None:
        return facts
    path = Path(root)
    facts["project_root"] = safe_name(path.name or str(path))
    try:
        entries = sorted(path.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return facts
    names = [safe_name(p.name) + ("/" if p.is_dir() else "") for p in entries if p.name != ".git"]
    shown = names[:MAX_TOP_LEVEL]
    extra = f" (+{len(names) - len(shown)} more)" if len(names) > len(shown) else ""
    facts["top_level"] = ", ".join(shown) + extra
    runners = [label for marker, label in _RUNNER_MARKERS if (path / marker).exists()]
    if runners:
        facts["toolchains"] = ", ".join(dict.fromkeys(runners))
    facts["git"] = "yes" if (path / ".git").exists() else "no"
    return facts


def build_system_prompt(facts: Mapping[str, str] | None = None) -> str:
    if not facts:
        return INSTRUCTIONS
    lines = [f"- {_clean(str(key))}: {_clean(str(value))}" for key, value in facts.items()]
    return INSTRUCTIONS + "\n\nProject facts:\n" + "\n".join(lines)
