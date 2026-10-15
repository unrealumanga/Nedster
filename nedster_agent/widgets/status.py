"""One-line status bar: model · context · speed · step · sandbox/isolation · elapsed · state."""

from __future__ import annotations

import time
from typing import Any

from rich.console import RenderableType
from rich.text import Text
from textual.widget import Widget

from .glyphs import Glyphs

STATES = ("idle", "starting", "thinking", "running tool", "cancelling…", "setup needed")
#: Unknown metrics are shown with this marker, never with an invented number.
UNKNOWN = "—"


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
        self.prompt_tokens: int | None = None  # latest prompt-token count of a model call, None = unknown
        self.decode_tps: float | None = None  # measured decode speed of the last model call, None = unknown
        self.step: int | None = None  # model calls in the active run, None while no run is active

    def set_info(self, info: dict[str, Any]) -> None:
        self.info = dict(info)
        self.refresh()

    def set_state(self, state: str, detail: str = "") -> None:
        self.state = state
        self.detail = detail
        self.refresh()

    def note_usage(self, prompt_tokens: int, completion_tokens: int, *, decode_ms: float = 0.0,
                   wall_s: float | None = None) -> None:
        """Record one model call: context fill (prompt tokens) and, when measurable, decode speed."""
        self.prompt_tokens = int(prompt_tokens) if int(prompt_tokens) > 0 else None
        seconds = decode_ms / 1000.0 if decode_ms and decode_ms > 0 else wall_s
        if seconds and seconds > 0 and completion_tokens > 0:
            self.decode_tps = completion_tokens / seconds
        self.refresh()

    def reset_usage(self) -> None:
        """Forget measured context/speed (after /resume or a model switch: they describe another session)."""
        self.prompt_tokens = None
        self.decode_tps = None
        self.refresh()

    def start_turn(self) -> None:
        self.turn_started = time.monotonic()
        self.last_elapsed = None
        self.step = 0
        self.refresh()

    def advance_step(self) -> None:
        self.step = (self.step or 0) + 1
        self.refresh()

    def end_turn(self) -> None:
        if self.turn_started is not None:
            self.last_elapsed = time.monotonic() - self.turn_started
        self.turn_started = None
        self.step = None
        self.refresh()

    @property
    def elapsed(self) -> float | None:
        if self.turn_started is not None:
            return time.monotonic() - self.turn_started
        return self.last_elapsed

    def render(self) -> RenderableType:
        """Left: model · context · speed · step · sandbox (truncated first). Right: elapsed · state · queue."""
        right = self._right()
        width = self.content_size.width or 120
        left = self._left()
        room = max(0, width - right.cell_len - 1)
        left.truncate(room, overflow="ellipsis")
        line = Text(no_wrap=True)
        line.append_text(left)
        line.append(" " * max(1, width - left.cell_len - right.cell_len))
        line.append_text(right)
        return line

    def _left(self) -> Text:
        g = self.glyphs
        info = self.info
        line = Text(no_wrap=True)
        line.append(g.text(str(info.get("model") or "no model")), style="bold")
        line.append(g.sep, style=g.style("faint"))
        line.append_text(self._context_metric(g))
        line.append(g.sep, style=g.style("faint"))
        line.append_text(self._speed_metric(g))
        line.append(g.sep, style=g.style("faint"))
        line.append_text(self._step_metric(g))
        label, danger = isolation_label(str(info.get("sandbox") or ""), str(info.get("isolation") or ""))
        line.append(g.sep, style=g.style("faint"))
        line.append(g.text(label), style=g.style("danger") if danger else g.style("muted"))
        return line

    def _context_metric(self, g: Glyphs) -> Text:
        """Percent of the context budget still free, from the latest prompt-token usage."""
        maximum = int(self.info.get("context_max") or 0)
        if self.prompt_tokens is None or maximum <= 0:
            return Text(g.text(f"ctx {UNKNOWN}"), style=g.style("faint"))
        used = min(self.prompt_tokens, maximum)
        style = g.style("fail" if used / maximum > 0.9 else "warn" if used / maximum > 0.7 else "muted")
        return Text(g.text(f"ctx {round(100 * (maximum - used) / maximum)}% left"), style=style)

    def _speed_metric(self, g: Glyphs) -> Text:
        if self.decode_tps is None:
            return Text(g.text(f"{UNKNOWN} tok/s"), style=g.style("faint"))
        return Text(g.text(f"{round(self.decode_tps)} tok/s"), style=g.style("muted"))

    def _step_metric(self, g: Glyphs) -> Text:
        if self.step is None:
            return Text(g.text(f"step {UNKNOWN}"), style=g.style("faint"))
        maximum = self.info.get("max_steps")
        text = f"step {self.step}/{int(maximum)}" if maximum else f"step {self.step}"
        return Text(g.text(text), style=g.style("info"))

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
