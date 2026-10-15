"""One-line status bar: model · context · sandbox/isolation · elapsed · state."""

from __future__ import annotations

import time
from typing import Any

from rich.console import RenderableType
from rich.text import Text
from textual.widget import Widget

from .glyphs import Glyphs

STATES = ("idle", "starting", "thinking", "running tool", "cancelling…", "setup needed")


def isolation_label(sandbox: str, isolation: str) -> tuple[str, bool]:
    """(label, dangerous). Host execution is always called out as unisolated."""
    if sandbox == "host" or isolation == "none":
        return "host — NO isolation", True
    if sandbox in {"", "unknown", "unavailable"}:
        return "no sandbox ready", True
    return f"{sandbox} {isolation}", False


class StatusBar(Widget):
    DEFAULT_CSS = "StatusBar { height: 1; padding: 0 2; }"

    def __init__(self, glyphs: Glyphs, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.glyphs = glyphs
        self.info: dict[str, Any] = {}
        self.state = "starting"
        self.detail = ""
        self.queued = 0
        self.turn_started: float | None = None
        self.last_elapsed: float | None = None

    def set_info(self, info: dict[str, Any]) -> None:
        self.info = dict(info)
        self.refresh()

    def set_state(self, state: str, detail: str = "") -> None:
        self.state = state
        self.detail = detail
        self.refresh()

    def start_turn(self) -> None:
        self.turn_started = time.monotonic()
        self.last_elapsed = None
        self.refresh()

    def end_turn(self) -> None:
        if self.turn_started is not None:
            self.last_elapsed = time.monotonic() - self.turn_started
        self.turn_started = None
        self.refresh()

    @property
    def elapsed(self) -> float | None:
        if self.turn_started is not None:
            return time.monotonic() - self.turn_started
        return self.last_elapsed

    def render(self) -> RenderableType:
        """Left: model · context · sandbox (truncated first). Right: elapsed · state · queue (never cut)."""
        right = self._right()
        width = self.content_size.width or 120
        left = self._left(compact=width < 100)
        room = max(0, width - right.cell_len - 1)
        left.truncate(room, overflow="ellipsis")
        line = Text(no_wrap=True)
        line.append_text(left)
        line.append(" " * max(1, width - left.cell_len - right.cell_len))
        line.append_text(right)
        return line

    def _left(self, *, compact: bool) -> Text:
        g = self.glyphs
        info = self.info
        line = Text(no_wrap=True)
        line.append(g.text(str(info.get("model") or "no model")), style="bold")
        role_label = str(info.get("role_label", "")).strip()
        if role_label:
            line.append(g.sep, style=g.style("faint"))
            line.append(g.text(role_label), style=g.style("muted"))
        used, maximum = int(info.get("context_used") or 0), int(info.get("context_max") or 0)
        line.append(g.sep, style=g.style("faint"))
        if maximum:
            ratio = used / maximum
            style = g.style("fail" if ratio > 0.9 else "warn" if ratio > 0.7 else "muted")
            bar = "" if compact else f"{g.bar(used, maximum, 8)} "
            line.append(f"ctx {bar}{_k(used)}/{_k(maximum)}", style=style)
        else:
            line.append("ctx n/a", style=g.style("faint"))
        label, danger = isolation_label(str(info.get("sandbox") or ""), str(info.get("isolation") or ""))
        line.append(g.sep, style=g.style("faint"))
        line.append(g.text(label), style=g.style("danger") if danger else g.style("muted"))
        return line

    def _right(self) -> Text:
        g = self.glyphs
        line = Text(no_wrap=True)
        elapsed = self.elapsed
        if elapsed is not None:
            line.append(f"{elapsed:.1f}s", style=g.style("muted"))
            line.append(g.sep, style=g.style("faint"))
        state_style = {"cancelling…": "warn", "setup needed": "fail", "idle": "ok"}.get(self.state, "info")
        line.append(g.text(self.state), style=g.style(state_style))
        if self.detail:
            line.append(g.text(f" {self.detail[:24]}"), style=g.style("muted"))
        if self.queued:
            line.append(g.text(f"{g.sep}{self.queued} queued"), style=g.style("warn"))
        return line


def _k(tokens: int) -> str:
    return f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)
