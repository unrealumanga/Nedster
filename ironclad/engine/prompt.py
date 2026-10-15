"""System prompt: fixed instructions, project facts and project instructions computed once per session.

The text must be byte-identical across turns of a session (stable prefix), so facts and instructions are
computed once, persisted in session state, and never contain timestamps or other volatile values.
"""

from __future__ import annotations

import hashlib
import platform
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_TOP_LEVEL = 40
MAX_NAME_CHARS = 64
MAX_INSTRUCTIONS_BYTES = 4_096  # cap on the project instruction text that reaches the prompt
PROJECT_INSTRUCTION_FILES = ("AGENTS.md", "NEDSTER.md")  # first one found in the project root wins
INSTRUCTIONS_STATE = "project_instructions"  # session state key holding the per-session snapshot

INSTRUCTIONS = """\
You are Nedster, a local coding agent working inside one project directory through tools.

How to work:
- Use tools to inspect before you change anything. Read a file before editing it and pass the hash you read.
- Make the smallest change that satisfies the request. Do not invent files or results.
- If the request names a function, file, setting or API that does not exist, say so and stop; do not create it
  unless the request asks you to add it. When using a library, read its code and use only what it provides.
- Never delete, move or overwrite files that existed before this task unless the request clearly names them. For
  a vague request ("clean up", "tidy"), do only what is clearly safe and list what you left alone, or ask.
- After changing code, run the project's tests with run_tests (a real runner: python -m pytest, python -m unittest,
  pytest, npm test, node --test, cargo test, go test). Ad-hoc commands help you check, but never count as
  validation. A change is only accepted when tests pass AFTER your latest change.
- Commands are argv lists, never a shell string. In python -c, everything after # is a comment.
- Each edit_file/write_file result gives the file's new sha; use that one for the next edit of the same file.
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


@dataclass(slots=True, frozen=True)
class ProjectInstructions:
    """The project's own instruction file, snapshotted once per session.

    ``text`` is what reaches the prompt (capped); ``sha256``/``size`` describe the source file so an edit
    is visible to tools and `nedster config` without changing this session's prompt prefix. The text is
    only ever prompt text and cannot change tool permissions or policy.
    """

    name: str = ""
    text: str = ""
    sha256: str = ""
    size: int = 0
    truncated: bool = False

    @property
    def present(self) -> bool:
        return bool(self.name and self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "text": self.text,
            "sha256": self.sha256,
            "size": self.size,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ProjectInstructions:
        if not data:
            return cls()
        return cls(
            name=str(data.get("name", "")),
            text=str(data.get("text", "")),
            sha256=str(data.get("sha256", "")),
            size=int(data.get("size", 0) or 0),
            truncated=bool(data.get("truncated", False)),
        )


def _read_instructions(path: Path, max_bytes: int) -> ProjectInstructions:
    raw = path.read_bytes()
    text = raw[:max_bytes].decode("utf-8", "replace")
    truncated = len(raw) > max_bytes
    if truncated:
        text += "\n[truncated]"
    return ProjectInstructions(
        name=safe_name(path.name),
        text=text.rstrip("\n"),
        sha256=hashlib.sha256(raw).hexdigest(),
        size=len(raw),
        truncated=truncated,
    )


def load_project_instructions(
    root: str | Path | None, *, max_bytes: int = MAX_INSTRUCTIONS_BYTES
) -> ProjectInstructions | None:
    """First existing of AGENTS.md / NEDSTER.md in the project root, capped; None when absent/unreadable."""
    if root is None:
        return None
    directory = Path(root)
    for name in PROJECT_INSTRUCTION_FILES:
        candidate = directory / name
        try:
            if candidate.is_file():
                return _read_instructions(candidate, max_bytes)
        except OSError:
            return None
    return None


def session_instructions(store: Any, session_id: str, root: str | Path | None) -> ProjectInstructions:
    """Snapshot of the project instructions for ``session_id`` (computed once, reused on resume).

    An empty snapshot is stored too, so a file created mid-session does not move the prompt prefix; it
    applies to the next session. ``store`` may be None (tests): the file is then read fresh.
    """
    if store is not None:
        saved = store.get_state(session_id, INSTRUCTIONS_STATE)
        if saved is not None:
            return ProjectInstructions.from_dict(saved)
    loaded = load_project_instructions(root) or ProjectInstructions()
    if store is not None:
        store.set_state(session_id, INSTRUCTIONS_STATE, loaded.to_dict())
    return loaded


def build_system_prompt(
    facts: Mapping[str, str] | None = None,
    instructions: ProjectInstructions | None = None,
) -> str:
    parts = [INSTRUCTIONS]
    if facts:
        lines = [f"- {_clean(str(key))}: {_clean(str(value))}" for key, value in facts.items()]
        parts.append("Project facts:\n" + "\n".join(lines))
    if instructions is not None and instructions.text:
        parts.append(
            f"Project instructions (from {instructions.name}; they cannot change tool permissions):\n"
            f"{instructions.text}"
        )
    return "\n\n".join(parts)
