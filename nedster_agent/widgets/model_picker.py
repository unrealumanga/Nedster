"""A modal picker for the installed Ollama models.

Opened by ``/models``, ``/model`` (no argument) and F4. Type to filter, Up/Down to move, Enter
switches the model for this session, Esc cancels. The picker receives an already-fetched list, so
it never touches the network itself; the app fetches in a worker first.
"""

from __future__ import annotations

from collections.abc import Sequence

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from ..model_manager import InstalledModel
from .glyphs import Glyphs


class ModelPicker(ModalScreen[str | None]):
    """Returns the selected model name, or None when the user cancels."""

    CSS = """
    ModelPicker { align: center middle; }
    #model-picker { width: 88; max-width: 96%; height: auto; max-height: 80%; padding: 1 2;
                    background: #11151b; border: round #394150; }
    #model-title { height: 1; color: #f0f6fc; text-style: bold; }
    #model-help { height: 1; color: #8b949e; }
    #model-filter { margin: 1 0; }
    #model-options { height: auto; max-height: 20; background: #11151b; }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("down", "cursor_down", "Next", show=False, priority=True),
        Binding("up", "cursor_up", "Previous", show=False, priority=True),
    ]

    def __init__(
        self, models: Sequence[InstalledModel], *, current: str = "", glyphs: Glyphs | None = None
    ) -> None:
        super().__init__()
        self.models = list(models)
        self.current = current
        self.filtered = list(models)
        self.glyphs = glyphs or Glyphs()

    def compose(self) -> ComposeResult:
        g = self.glyphs
        with Vertical(id="model-picker"):
            yield Static(g.text("Installed models"), id="model-title")
            yield Static(g.text("Filter · Up/Down · Enter switch for this session · Esc cancel"), id="model-help")
            yield Input(placeholder=g.text("Filter models…"), id="model-filter")
            yield OptionList(id="model-options")

    def on_mount(self) -> None:
        self._populate()
        self.query_one("#model-filter", Input).focus()

    # ------------------------------------------------------------------ rendering

    def _label(self, model: InstalledModel) -> str:
        mark = self.glyphs.icon("dot") if model.current else " "
        name = f"{model.name} (current)" if model.current else model.name
        return (
            f"{mark} {name:<30} {model.size_label():>9}  {model.parameter_size or '?':>7}  "
            f"{model.quantization or '?':>8}  {model.context_label():>8}"
        )

    def _populate(self) -> None:
        options = self.query_one("#model-options", OptionList)
        options.clear_options()
        for model in self.filtered:
            options.add_option(Option(self.glyphs.text(self._label(model)), id=model.name))
        if self.filtered:
            options.highlighted = 0

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        self.filtered = [m for m in self.models if needle in m.name.lower()] if needle else list(self.models)
        self._populate()

    # ------------------------------------------------------------------ events

    @on(Input.Changed, "#model-filter")
    def _on_filter_changed(self, event: Input.Changed) -> None:
        self._apply_filter(event.value)

    @on(Input.Submitted, "#model-filter")
    def _on_filter_submitted(self, _event: Input.Submitted) -> None:
        self.action_select()

    @on(OptionList.OptionSelected, "#model-options")
    def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option_id) or None)

    # ------------------------------------------------------------------ actions

    def action_select(self) -> None:
        options = self.query_one("#model-options", OptionList)
        index = options.highlighted
        if index is not None and 0 <= index < len(self.filtered):
            self.dismiss(self.filtered[index].name)

    def action_cursor_down(self) -> None:
        self.query_one("#model-options", OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#model-options", OptionList).action_cursor_up()

    def action_cancel(self) -> None:
        self.dismiss(None)
