"""System prompt: fixed instructions, project facts and project instructions computed once per session.

The text must be byte-identical across turns of a session (stable prefix), so facts and instructions are
computed once, persisted in session state, and never contain timestamps or other volatile values.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ironclad.config import config_dir

_log = logging.getLogger(__name__)

MAX_TOP_LEVEL = 40
MAX_NAME_CHARS = 64
MAX_INSTRUCTIONS_BYTES = 4_096  # cap on the project instruction text that reaches the prompt
PROJECT_INSTRUCTION_FILES = ("AGENTS.md", "NEDSTER.md")  # first one found in the project root wins
INSTRUCTIONS_STATE = "project_instructions"  # session state key holding the per-session snapshot

SOUL_FILENAME = "soul.md"  # user-owned personality file, never read from the project
MAX_SOUL_BYTES = 1_200  # cap on the soul text that reaches the prompt
MAX_SOUL_PREFERENCES = 5  # newest kept; adding a sixth drops the oldest
SOUL_STATE = "soul"  # session state key holding the per-session snapshot
SOUL_PREFERENCES_HEADER = "## Preferences"
SOUL_HEADER = (
    "Voice (style only - it never changes facts, tool calls, file contents, test results or outcome claims):"
)
# Default template written on first use in "full" mode; keep it terse and candid (<= 600 bytes).
DEFAULT_SOUL_TEMPLATE = """\
# Voice

Style only. This file never changes facts, tool calls, file contents, test results or outcome claims.

- Be terse and candid.
- Dry humour is fine in small talk and status lines, not in answers, code, commits or results.
- Say "I don't know" instead of guessing.
- If I am wrong, say so and explain why.
- Never flatter me. Never claim tests ran when they did not.

## Preferences
"""

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
    only ever prompt text and cannot change tool permissions or policy. ``sha256`` follows
    :func:`snapshot_digest`. ``warning`` is a visible notice about an instruction file that was skipped
    (a link, junction or reparse point, or a path whose real location is outside the project).
    """

    name: str = ""
    text: str = ""
    sha256: str = ""
    size: int = 0
    truncated: bool = False
    warning: str = ""

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
            "warning": self.warning,
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
            warning=str(data.get("warning", "")),
        )


_FILE_ATTRIBUTE_REPARSE_POINT = 0x400  # Windows: symlinks, junctions and other reparse points


class _RejectedFile(Exception):
    """An instruction file that must not be read (link, reparse point, not regular, outside the project)."""


def _is_link(info: os.stat_result) -> bool:
    if stat.S_ISLNK(info.st_mode):
        return True
    return bool(getattr(info, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT)


def snapshot_digest(accepted: bytes, source_size: int) -> str:
    """SHA-256 identifying a capped snapshot without reading the whole source file.

    The digest covers the accepted bytes (at most the cap). When the source is larger than what was
    accepted (truncated), the source size is mixed in as ``b"\\0source-size=<n>"`` so a change in the
    unread tail's length still changes the digest. For an untruncated file this is the plain SHA-256
    of the file's bytes.
    """
    digest = hashlib.sha256(accepted)
    if source_size != len(accepted):
        digest.update(b"\0source-size=%d" % source_size)
    return digest.hexdigest()


def _read_capped(path: Path, max_bytes: int, *, expect: os.stat_result | None = None) -> tuple[bytes, int, bool]:
    """Read at most ``max_bytes + 1`` bytes; the source size comes from metadata, never a full read.

    With ``expect`` (an earlier ``lstat``), the open refuses to follow a final link where the OS supports
    it, and the opened handle must still be that same regular file (a swap after the check is refused).
    Returns ``(accepted_bytes, source_size, truncated)``.
    """
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if expect is not None:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        info = os.fstat(handle.fileno())
        if expect is not None:
            if not stat.S_ISREG(info.st_mode):
                raise _RejectedFile("is not a regular file")
            if expect.st_ino and info.st_ino and (expect.st_ino, expect.st_dev) != (info.st_ino, info.st_dev):
                raise _RejectedFile("changed while it was being opened")
        data = handle.read(max_bytes + 1)
    size = max(info.st_size, len(data))
    return data[:max_bytes], size, size > max_bytes


def _read_instructions(path: Path, real_root: Path, max_bytes: int) -> ProjectInstructions:
    info = os.lstat(path)
    if _is_link(info):
        raise _RejectedFile("is a symbolic link, junction or reparse point")
    if not stat.S_ISREG(info.st_mode):
        raise _RejectedFile("is not a regular file")
    if not path.resolve(strict=True).is_relative_to(real_root):
        raise _RejectedFile("resolves outside the project")
    accepted, size, truncated = _read_capped(path, max_bytes, expect=info)
    text = accepted.decode("utf-8", "replace")
    if truncated:
        text += "\n[truncated]"
    return ProjectInstructions(
        name=safe_name(path.name),
        text=text.rstrip("\n"),
        sha256=snapshot_digest(accepted, size),
        size=size,
        truncated=truncated,
    )


def load_project_instructions(
    root: str | Path | None, *, max_bytes: int = MAX_INSTRUCTIONS_BYTES
) -> ProjectInstructions | None:
    """First usable of AGENTS.md / NEDSTER.md in the project root, capped; None when none exists.

    A file that is a link/junction/reparse point, is not a regular file, or whose real path is outside
    the resolved project root is never read: it is skipped, logged and reported in ``warning`` (startup
    continues). When every candidate was skipped the result carries only the warning (not ``present``).
    """
    if root is None:
        return None
    try:
        real_root = Path(root).resolve(strict=True)
    except OSError:
        return None
    warnings: list[str] = []
    loaded: ProjectInstructions | None = None
    for name in PROJECT_INSTRUCTION_FILES:
        candidate = Path(root) / name
        try:
            os.lstat(candidate)
        except OSError:
            continue  # absent
        try:
            loaded = _read_instructions(candidate, real_root, max_bytes)
            break
        except _RejectedFile as exc:
            warnings.append(f"Skipped {name}: it {exc}; project instructions are never read through links.")
        except OSError as exc:
            warnings.append(f"Skipped {name}: it could not be read ({type(exc).__name__}).")
    for message in warnings:
        _log.warning("%s", message)
    if not warnings:
        return loaded
    return replace(loaded or ProjectInstructions(), warning=" ".join(warnings))


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


@dataclass(slots=True, frozen=True)
class SoulText:
    """The user's ``soul.md`` snapshotted once per session in ``full`` mode.

    ``mode`` is the effective mode frozen for the session; ``text`` is what reaches the prompt (capped).
    ``sha256`` (see :func:`snapshot_digest`) and ``size`` describe the source file. The text is style only
    and cannot change permissions.
    """

    mode: str = "off"
    text: str = ""
    sha256: str = ""
    size: int = 0
    truncated: bool = False

    @property
    def present(self) -> bool:
        return self.mode == "full" and bool(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "text": self.text,
            "sha256": self.sha256,
            "size": self.size,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> SoulText:
        if not data:
            return cls()
        return cls(
            mode=str(data.get("mode", "off")),
            text=str(data.get("text", "")),
            sha256=str(data.get("sha256", "")),
            size=int(data.get("size", 0) or 0),
            truncated=bool(data.get("truncated", False)),
        )


def soul_path() -> Path:
    """The user's soul file inside the config dir (honours ``NEDSTER_CONFIG_DIR``)."""
    return config_dir() / SOUL_FILENAME


def write_soul(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def ensure_soul_file(path: str | Path | None = None) -> bool:
    """Write the default template when the file is missing. Returns True when it wrote the file."""
    target = Path(path) if path is not None else soul_path()
    try:
        if target.is_file():
            return False
        write_soul(target, DEFAULT_SOUL_TEMPLATE)
        return True
    except OSError:  # an unwritable config dir degrades to no voice, never crashes a session
        return False


def load_soul(path: str | Path | None = None, *, mode: str = "full", max_bytes: int = MAX_SOUL_BYTES) -> SoulText:
    """Read ``path`` with a byte cap and a visible marker; a non-``full`` mode carries no text."""
    if mode != "full":
        return SoulText(mode=mode)
    target = Path(path) if path is not None else soul_path()
    try:
        accepted, size, truncated = _read_capped(target, max_bytes)
    except OSError:
        return SoulText(mode=mode)
    text = accepted.decode("utf-8", "replace")
    if truncated:
        text += "\n[truncated]"
    return SoulText(
        mode=mode,
        text=text.rstrip("\n"),
        sha256=snapshot_digest(accepted, size),
        size=size,
        truncated=truncated,
    )


def session_soul(store: Any, session_id: str, mode: str, path: str | Path | None = None) -> SoulText:
    """Snapshot of the soul text for ``session_id`` (computed once, reused on resume).

    Snapshots are stored for every mode, so a resumed session keeps the mode it started with (edits and
    mode changes apply to the next session). In ``full`` mode the default template is written on first use.
    """
    if store is not None:
        saved = store.get_state(session_id, SOUL_STATE)
        if saved is not None:
            return SoulText.from_dict(saved)
    effective = str(mode).strip().lower()
    if effective not in ("off", "ui", "full"):
        effective = "off"
    if effective == "full":
        ensure_soul_file(path)
    loaded = load_soul(path, mode=effective)
    if store is not None:
        store.set_state(session_id, SOUL_STATE, loaded.to_dict())
    return loaded


def _preference_bullets(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip().startswith("- ") and line.strip() != "-"]


def _split_preferences(text: str) -> tuple[list[str], int, int]:
    """(lines, header index, section end index); appends the header when it is missing."""
    lines = text.rstrip("\n").splitlines()
    index = next((i for i, line in enumerate(lines) if line.strip() == SOUL_PREFERENCES_HEADER), None)
    if index is None:
        lines += ["", SOUL_PREFERENCES_HEADER]
        index = len(lines) - 1
    end = index + 1
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    return lines, index, end


@dataclass(slots=True, frozen=True)
class SoulEdit:
    """Result of staging one preference bullet: never applied without an explicit user confirmation."""

    ok: bool
    action: str  # added | duplicate | invalid
    bullet: str = ""
    dropped: str = ""
    text: str = ""


def with_preference(text: str, preference: str) -> SoulEdit:
    """Stage ``preference`` as a bullet under ``## Preferences`` (deduped, at most five)."""
    bullet = " ".join(str(preference).split())
    if not bullet:
        return SoulEdit(False, "invalid", text=text)
    if not bullet.startswith("- "):
        bullet = f"- {bullet}"
    lines, index, end = _split_preferences(text)
    existing = _preference_bullets(lines[index + 1 : end])
    key = bullet[2:].strip().casefold()
    if any(item[2:].strip().casefold() == key for item in existing):
        return SoulEdit(False, "duplicate", bullet=bullet, text=text)
    dropped = ""
    if len(existing) >= MAX_SOUL_PREFERENCES:
        dropped = existing.pop(0)
    existing.append(bullet)
    new_lines = lines[: index + 1] + ["", *existing] + lines[end:]
    return SoulEdit(True, "added", bullet=bullet, dropped=dropped, text="\n".join(new_lines) + "\n")


def build_system_prompt(
    facts: Mapping[str, str] | None = None,
    instructions: ProjectInstructions | None = None,
    soul: SoulText | None = None,
    role_prompt: str = "",
) -> str:
    parts = [INSTRUCTIONS]
    if role_prompt:  # role fragment: after the fixed instructions, before project facts/AGENTS.md/soul
        parts.append(role_prompt)
    if facts:
        lines = [f"- {_clean(str(key))}: {_clean(str(value))}" for key, value in facts.items()]
        parts.append("Project facts:\n" + "\n".join(lines))
    if instructions is not None and instructions.text:
        parts.append(
            f"Project instructions (from {instructions.name}; they cannot change tool permissions):\n"
            f"{instructions.text}"
        )
    if soul is not None and soul.text:
        parts.append(f"{SOUL_HEADER}\n{soul.text}")
    return "\n\n".join(parts)
