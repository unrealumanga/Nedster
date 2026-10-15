"""A scrollable modal showing one tool call's complete stored output.

Opened from a tool row (press ``o``). The text is loaded lazily in a worker through ``loader`` and
capped by :func:`render_full_output` so a huge output never freezes the interface. Esc closes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import RichLog, Static

from .glyphs import Glyphs
from .transcript import render_full_output


class OutputViewer(ModalScreen[None]):
    """Read-only viewer for a stored tool output; returns None on close."""

    CSS = """
    OutputViewer { align: center middle; }
    #output-viewer { width: 96%; height: 88%; padding: 1 2; background: #11151b; border: round #394150; }
    #output-title { height: 1; color: #f0f6fc; text-style: bold; }
    #output-note { height: 1; color: #8b949e; }
    #output-body { height: 1fr; background: #11151b; }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Close", show=False, priority=True),
        Binding("q", "cancel", "Close", show=False),
    ]

    def __init__(
        self,
        glyphs: Glyphs,
        *,
        title: str = "tool",
        loader: Callable[[], str] | None = None,
        fallback: str = "",
        historical: bool = False,
    ) -> None:
        super().__init__()
        self.glyphs = glyphs
        self.title_text = title
        self.loader = loader
        self.fallback = fallback
        self.historical = historical

    def compose(self) -> ComposeResult:
        g = self.glyphs
        with Vertical(id="output-viewer"):
            yield Static(g.text(f"Full output · {self.title_text}"), id="output-title")
            yield Static(g.text("Esc closes"), id="output-note")
            yield RichLog(id="output-body", markup=False, wrap=True, auto_scroll=False, highlight=False)

    def on_mount(self) -> None:
        self.query_one("#output-body", RichLog).write(self.glyphs.text("Loading…"))
        self.run_worker(self._load(), group="output", exclusive=True)

    async def _load(self) -> None:
        text = ""
        if self.loader is not None:
            text = await asyncio.to_thread(self.loader)
        if not text:
            text = self.fallback
        body = self.query_one("#output-body", RichLog)
        body.clear()
        if not text:
            body.write(self.glyphs.text("(no stored output for this call)"))
            return
        body.write(self.glyphs.text(render_full_output(text)))
        note = self.query_one("#output-note", Static)
        label = "Historical output · not current evidence · Esc closes" if self.historical else "Esc closes"
        note.update(self.glyphs.text(label))

    def action_cancel(self) -> None:
        self.dismiss(None)
