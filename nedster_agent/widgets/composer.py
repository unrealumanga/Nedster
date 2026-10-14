"""Multi-line composer with persisted history and @file / slash-command completion."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from textual import events
from textual.message import Message
from textual.widgets import OptionList, TextArea

HISTORY_LIMIT = 500
FILE_LIMIT = 20_000
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
             ".pytest_cache", ".ruff_cache", "dist", "build", ".tox", ".idea"}
_MENTION = re.compile(r"(?:^|\s)@([^\s@]*)$")


class History:
    """Prompt history persisted per project as JSON lines (multi-line entries survive)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.entries: list[str] = []
        self.index: int | None = None
        self.draft = ""
        if path is not None and path.exists():
            try:
                with path.open(encoding="utf-8") as handle:
                    self.entries = [str(json.loads(line)) for line in handle if line.strip()]
            except (OSError, ValueError):
                self.entries = []
        self.entries = self.entries[-HISTORY_LIMIT:]

    @staticmethod
    def for_project(data_dir: Path, project_root: Path) -> History:
        key = hashlib.sha256(str(project_root.resolve()).encode("utf-8")).hexdigest()[:16]
        return History(Path(data_dir) / "history" / f"{key}.jsonl")

    def add(self, text: str) -> None:
        self.index = None
        if not text.strip() or (self.entries and self.entries[-1] == text):
            return
        self.entries.append(text)
        if len(self.entries) > HISTORY_LIMIT:
            del self.entries[: len(self.entries) - HISTORY_LIMIT]
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(e) + "\n" for e in self.entries), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass  # history is a convenience; never break the composer

    def previous(self, current: str) -> str | None:
        if not self.entries:
            return None
        if self.index is None:
            self.draft = current
            self.index = len(self.entries)
        if self.index == 0:
            return None
        self.index -= 1
        return self.entries[self.index]

    def next(self) -> str | None:
        if self.index is None:
            return None
        self.index += 1
        if self.index >= len(self.entries):
            self.index = None
            return self.draft
        return self.entries[self.index]


def project_files(root: Path) -> list[str]:
    """Tracked + untracked-not-ignored files via git; os.walk fallback outside a repository."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root, capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            return sorted({line for line in result.stdout.splitlines() if line})[:FILE_LIMIT]
    except (OSError, subprocess.SubprocessError):
        pass
    found: list[str] = []
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        base = Path(current).relative_to(root)
        for name in sorted(files):
            found.append((base / name).as_posix())
            if len(found) >= FILE_LIMIT:
                return found
    return found


def match_files(files: Sequence[str], prefix: str, limit: int = 12) -> list[str]:
    needle = prefix.lower()
    starts = [f for f in files if f.lower().startswith(needle)]
    names = [f for f in files if f not in starts and Path(f).name.lower().startswith(needle)]
    contains = [f for f in files if f not in starts and f not in names and needle in f.lower()]
    return (starts + names + contains)[:limit]


@dataclass(slots=True, frozen=True)
class Completion:
    kind: str  # "file" | "command"
    start: int  # column in the cursor's line where the replaced token begins
    items: tuple[str, ...]


class Composer(TextArea):
    """Enter submits; Shift+Enter / Ctrl+J insert a newline; pastes never submit."""

    DEFAULT_CSS = "Composer { height: auto; min-height: 3; max-height: 8; }"

    class Submitted(Message):
        def __init__(self, composer: Composer, value: str) -> None:
            super().__init__()
            self.composer = composer
            self.value = value

        @property
        def control(self) -> Composer:
            return self.composer

    def __init__(
        self,
        *,
        history: History,
        commands: Sequence[str],
        files: Callable[[], Sequence[str]],
        menu: Callable[[], OptionList],
        id: str | None = None,
    ) -> None:
        super().__init__(id=id, soft_wrap=True, tab_behavior="focus", show_line_numbers=False)
        self.prompt_history = history
        self.commands = list(commands)
        self._files_source = files
        self._files: Sequence[str] | None = None
        self._menu = menu
        self.completion: Completion | None = None
        self._suppress_completion = False

    # ------------------------------------------------------------------ keys

    async def _on_key(self, event: events.Key) -> None:
        key = event.key
        if event.character == "?" and not self.text:
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, "/help"))
            return
        if key in {"shift+enter", "ctrl+j"}:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if key == "enter":
            event.stop()
            event.prevent_default()
            if self.completion is not None:
                self.accept_completion()
                return
            self.post_message(self.Submitted(self, self.text))
            return
        if key == "tab" and self.completion is not None:
            event.stop()
            event.prevent_default()
            self.accept_completion()
            return
        await super()._on_key(event)

    def action_cursor_up(self, select: bool = False) -> None:
        if not select and self.completion is not None:
            self._menu().action_cursor_up()
            return
        if not select and self.cursor_location[0] == 0:
            previous = self.prompt_history.previous(self.text)
            if previous is not None:
                self._replace_all(previous)
            return
        super().action_cursor_up(select)

    def action_cursor_down(self, select: bool = False) -> None:
        if not select and self.completion is not None:
            self._menu().action_cursor_down()
            return
        if not select and self.cursor_location[0] == self.document.line_count - 1 and self.prompt_history.index is not None:
            following = self.prompt_history.next()
            if following is not None:
                self._replace_all(following)
            return
        super().action_cursor_down(select)

    def _replace_all(self, text: str) -> None:
        self._suppress_completion = True
        self.load_text(text)
        last = self.document.line_count - 1
        self.move_cursor((last, len(self.document[last])))
        self.close_completion()

    def set_text(self, text: str) -> None:
        self._replace_all(text)

    # ------------------------------------------------------------------ completion

    def files(self) -> Sequence[str]:
        if self._files is None:
            self._files = self._files_source()
        return self._files

    def _on_text_area_changed(self, event: TextArea.Changed) -> None:
        if self._suppress_completion:
            self._suppress_completion = False
            return
        self.update_completion()

    def update_completion(self) -> None:
        row, column = self.cursor_location
        before = self.document[row][:column]
        found: Completion | None = None
        if row == 0 and before.startswith("/") and " " not in before:
            items = tuple(c for c in self.commands if c.startswith(before.lower()))
            if items and items != (before,):
                found = Completion("command", 0, items)
        else:
            mention = _MENTION.search(before)
            if mention:
                items = tuple(match_files(self.files(), mention.group(1)))
                if items:
                    found = Completion("file", mention.start(1) - 1, items)
        self.completion = found
        menu = self._menu()
        if found is None:
            menu.display = False
            return
        menu.clear_options()
        menu.add_options(list(found.items))
        menu.highlighted = 0
        menu.display = True

    def close_completion(self) -> None:
        self.completion = None
        self._menu().display = False

    def accept_completion(self, index: int | None = None) -> None:
        completion = self.completion
        if completion is None:
            return
        menu = self._menu()
        chosen = completion.items[index if index is not None else (menu.highlighted or 0)]
        row, column = self.cursor_location
        replacement = f"@{chosen} " if completion.kind == "file" else f"{chosen} "
        self.replace(replacement, (row, completion.start), (row, column))
        self.close_completion()
