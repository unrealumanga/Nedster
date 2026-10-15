"""The delete-confirmation modal.

Mirrors the CLI preview (title, last update, message and checkpoint counts) and asks for an
explicit ``y``/``yes`` before a conversation and its saved checkpoints are removed. It never
deletes by itself: the app runs the store delete and dismisses with ``True`` only on confirm.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

PROMPT = "Delete this conversation and its saved checkpoints? [y/N]"


class DeleteConfirm(ModalScreen[bool]):
    """Dismisses with ``True`` on confirm, ``False`` on cancel."""

    BINDINGS = [
        Binding("y", "confirm", "Delete"),
        Binding("n", "cancel", "Cancel"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    DeleteConfirm { align: center middle; background: #000000 60%; }
    #delete-box { width: 80%; max-width: 90; height: auto; background: #0b0d10; color: #e6edf3;
                  border: round #f7768e; padding: 1 2; }
    #delete-head { height: 1; color: #f0f6fc; text-style: bold; }
    #delete-body { height: auto; margin: 1 0; color: #8b949e; }
    #delete-question { height: auto; color: #f0f6fc; }
    #delete-help { height: 1; margin-top: 1; color: #6e7681; }
    .no-color #delete-box, .no-color #delete-head, .no-color #delete-body,
    .no-color #delete-question, .no-color #delete-help {
        color: ansi_default; background: ansi_default; border: round ansi_default; }
    .ascii #delete-box { border: ascii; }
    """

    def __init__(self, preview: str, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._preview = preview

    def compose(self) -> ComposeResult:
        with Vertical(id="delete-box"):
            yield Static("Delete conversation", id="delete-head")
            yield Static(self._preview, id="delete-body", markup=False)
            yield Static(PROMPT, id="delete-question", markup=False)
            yield Static("Y delete · N/Esc cancel", id="delete-help", markup=False)

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


__all__ = ["DeleteConfirm", "PROMPT"]
