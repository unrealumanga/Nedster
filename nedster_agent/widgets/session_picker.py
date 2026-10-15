"""The searchable conversation picker.

A modal over the transcript that lists every saved conversation of the project (never only the
newest page), filters as you type, previews the first and last user prompt, and resumes on Enter.
Locked rows are conversations another live Nedster is already using; they are shown but cannot be
chosen. While a turn is running the picker is read-only, so a switch can never race the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from ironclad.session_store.titles import display_title, relative_age

SHORT_ID = 8
PREVIEW_CHARS = 160


@dataclass(slots=True, frozen=True)
class SessionOption:
    """One row of the picker; built from a store listing (see ``search_sessions``)."""

    id: str
    title: str = ""
    message_count: int = 0
    updated_ns: int = 0
    first_user_text: str = ""
    last_user_text: str = ""
    current: bool = False
    locked: bool = False


def _clip(text: str, limit: int = PREVIEW_CHARS) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _matches(option: SessionOption, needle: str) -> bool:
    if not needle:
        return True
    if option.id.lower().startswith(needle):
        return True
    if needle in option.title.lower():
        return True
    return needle in option.first_user_text.lower()


class SessionPicker(ModalScreen[str | None]):
    """Dismisses with the chosen session id, or ``None`` when cancelled."""

    BINDINGS = [Binding("escape", "cancel", "Close")]

    CSS = """
    SessionPicker { align: center middle; background: #000000 60%; }
    #session-picker { width: 90%; max-width: 110; height: auto; max-height: 90%;
                      background: #0b0d10; color: #e6edf3; border: round #7aa2f7; padding: 1 2; }
    #session-head { height: 1; color: #f0f6fc; text-style: bold; }
    #session-filter { height: 3; margin: 1 0; background: #11151b; color: #f0f6fc; border: round #394150; }
    #session-filter:focus { border: round #7aa2f7; }
    #session-rows { height: auto; max-height: 14; background: #0b0d10; }
    #session-preview { height: auto; max-height: 6; margin-top: 1; color: #8b949e; }
    #session-help { height: 1; color: #6e7681; }
    .no-color #session-picker, .no-color #session-head, .no-color #session-filter,
    .no-color #session-rows, .no-color #session-preview, .no-color #session-help {
        color: ansi_default; background: ansi_default; border: round ansi_default; }
    .ascii #session-picker, .ascii #session-filter { border: ascii; }
    """

    def __init__(
        self,
        options: list[SessionOption],
        *,
        busy: bool = False,
        ascii_only: bool = False,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._all = list(options)
        self._busy = busy
        self._marker = "*" if ascii_only else "•"
        self._filtered: list[SessionOption] = []
        self._query = ""

    # ---------------------------------------------------------------- layout

    def compose(self) -> ComposeResult:
        with Vertical(id="session-picker"):
            yield Static("Saved conversations", id="session-head")
            yield Input(placeholder="Type to filter by name, first message or id", id="session-filter")
            yield OptionList(id="session-rows")
            yield Static("", id="session-preview")
            yield Static("", id="session-help")

    def on_mount(self) -> None:
        self._set_help()
        self._rebuild()
        self.query_one("#session-filter", Input).focus()

    # ---------------------------------------------------------------- state

    def _set_help(self) -> None:
        help_text = (
            "Finish or stop the current turn to switch conversations."
            if self._busy
            else "Enter resume · Up/Down move · Esc close (your draft is kept)"
        )
        self.query_one("#session-help", Static).update(help_text)
        if self._busy:
            self.query_one("#session-head", Static).update("Saved conversations  (read-only while a turn runs)")

    def _label(self, option: SessionOption) -> str:
        marker = self._marker if option.current else " "
        locked = "  [locked]" if option.locked else ""
        title = display_title(option.title, option.first_user_text)
        return (
            f" {marker} {option.id[:SHORT_ID]}  {relative_age(option.updated_ns):<10}"
            f"{option.message_count:>3} msg  {title}{locked}"
        )

    def _rebuild(self) -> None:
        needle = self._query.strip().lower()
        self._filtered = [option for option in self._all if _matches(option, needle)]
        rows = self.query_one("#session-rows", OptionList)
        rows.clear_options()
        rows.add_options([Option(self._label(option), id=option.id) for option in self._filtered])
        if self._filtered:
            rows.highlighted = 0
        self._update_preview()

    def _highlighted(self) -> SessionOption | None:
        rows = self.query_one("#session-rows", OptionList)
        index = rows.highlighted
        if index is None or not 0 <= index < len(self._filtered):
            return None
        return self._filtered[index]

    def _update_preview(self) -> None:
        preview = self.query_one("#session-preview", Static)
        option = self._highlighted()
        if option is None:
            preview.update("No matching conversations." if self._all else "No saved conversations for this project.")
            return
        first = _clip(option.first_user_text)
        last = _clip(option.last_user_text)
        lines = [f"{display_title(option.title, option.first_user_text)}  ·  {option.message_count} message(s)"]
        if first:
            lines.append(f"first: {first}")
        if last and last != first:
            lines.append(f"last:  {last}")
        preview.update("\n".join(lines))

    def _notice(self, text: str) -> None:
        self.query_one("#session-preview", Static).update(text)

    # ---------------------------------------------------------------- events

    def on_key(self, event: Any) -> None:
        rows = self.query_one("#session-rows", OptionList)
        if event.key == "down":
            rows.action_cursor_down()
        elif event.key == "up":
            rows.action_cursor_up()
        else:
            return
        event.prevent_default()
        event.stop()

    @on(Input.Changed, "#session-filter")
    def _filter_changed(self, event: Input.Changed) -> None:
        self._query = event.value
        self._rebuild()

    @on(Input.Submitted, "#session-filter")
    def _filter_submitted(self, _event: Input.Submitted) -> None:
        self._choose()

    @on(OptionList.OptionHighlighted, "#session-rows")
    def _highlighted_moved(self, _event: OptionList.OptionHighlighted) -> None:
        self._update_preview()

    @on(OptionList.OptionSelected, "#session-rows")
    def _option_selected(self, _event: OptionList.OptionSelected) -> None:
        self._choose()

    def _choose(self) -> None:
        if self._busy:
            self._notice("Finish or stop the current turn to switch conversations.")
            return
        option = self._highlighted()
        if option is None:
            return
        if option.locked:
            self._notice(
                f"Session {option.id[:SHORT_ID]} is active in another Nedster window; close it there first."
            )
            return
        self.dismiss(option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)
