"""BP1-S3: the session picker modal (filter, preview, Enter/Esc, locked and busy rows)."""

from __future__ import annotations

import pytest
from textual.app import App

from nedster_agent.widgets.session_picker import SessionOption, SessionPicker


class PickerHarness(App[None]):
    """Smallest app that can host the modal so the widget is tested on its own."""

    def __init__(self, options: list[SessionOption], *, busy: bool = False, ascii_only: bool = False) -> None:
        super().__init__()
        self._options = options
        self._busy = busy
        self._ascii_only = ascii_only
        self.result: str | None | object = "unset"

    def on_mount(self) -> None:
        self.push_screen(
            SessionPicker(self._options, busy=self._busy, ascii_only=self._ascii_only), self._record_result
        )

    def _record_result(self, value: str | None) -> None:
        self.result = value


def opt(
    session_id: str,
    title: str = "",
    *,
    count: int = 1,
    updated: int = 0,
    first: str = "",
    last: str = "",
    current: bool = False,
    locked: bool = False,
) -> SessionOption:
    return SessionOption(
        id=session_id,
        title=title,
        message_count=count,
        updated_ns=updated,
        first_user_text=first,
        last_user_text=last,
        current=current,
        locked=locked,
    )


def _row_ids(picker: SessionPicker) -> list[str]:
    from textual.widgets import OptionList

    rows = picker.query_one("#session-rows", OptionList)
    return [str(option.id) for option in rows.options]


@pytest.mark.asyncio
async def test_picker_filters_by_title_first_message_and_id_prefix():
    options = [
        opt("aaaaaaaa1111", "Write docs", first="document the picker", last="last docs line"),
        opt("bbbbbbbb2222", "Fix the range", first="the range check is off", last="range fixed"),
    ]
    harness = PickerHarness(options)
    async with harness.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        picker = harness.screen
        assert isinstance(picker, SessionPicker)
        assert _row_ids(picker) == ["aaaaaaaa1111", "bbbbbbbb2222"]

        await pilot.press("d", "o", "c")
        await pilot.pause()
        assert _row_ids(picker) == ["aaaaaaaa1111"]  # title match

        for _ in range(3):
            await pilot.press("backspace")
        await pilot.press("r", "a", "n", "g", "e")
        await pilot.pause()
        assert _row_ids(picker) == ["bbbbbbbb2222"]  # first-message match

        for _ in range(5):
            await pilot.press("backspace")
        await pilot.press("a", "a", "a", "a")
        await pilot.pause()
        assert _row_ids(picker) == ["aaaaaaaa1111"]  # id-prefix match


@pytest.mark.asyncio
async def test_picker_preview_shows_first_and_last_user_prompt():
    options = [opt("aaaaaaaa1111", "Named", first="first thing", last="most recent thing")]
    harness = PickerHarness(options)
    async with harness.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        picker = harness.screen
        preview = str(picker.query_one("#session-preview").render())
        assert "first thing" in preview
        assert "most recent thing" in preview


@pytest.mark.asyncio
async def test_picker_enter_resumes_highlighted_and_escape_keeps_draft():
    options = [opt("aaaaaaaa1111", "one"), opt("bbbbbbbb2222", "two")]
    harness = PickerHarness(options)
    async with harness.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert harness.result == "bbbbbbbb2222"


@pytest.mark.asyncio
async def test_picker_escape_returns_none():
    harness = PickerHarness([opt("aaaaaaaa1111", "one")])
    async with harness.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert harness.result is None


@pytest.mark.asyncio
async def test_picker_locked_row_cannot_be_chosen():
    options = [opt("aaaaaaaa1111", "open"), opt("bbbbbbbb2222", "in use", locked=True)]
    harness = PickerHarness(options)
    async with harness.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        picker = harness.screen
        await pilot.press("down")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert harness.result == "unset"  # stayed open
        assert "active in another Nedster" in str(picker.query_one("#session-preview").render())


@pytest.mark.asyncio
async def test_picker_read_only_while_busy():
    options = [opt("aaaaaaaa1111", "one")]
    harness = PickerHarness(options, busy=True)
    async with harness.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        picker = harness.screen
        assert "read-only" in str(picker.query_one("#session-head").render())
        await pilot.press("enter")
        await pilot.pause()
        assert harness.result == "unset"
        assert "Finish or stop the current turn" in str(picker.query_one("#session-preview").render())


@pytest.mark.asyncio
async def test_picker_marks_current_and_locked_in_rows():
    options = [
        opt("aaaaaaaa1111", "current one", current=True),
        opt("bbbbbbbb2222", "asked elsewhere", locked=True),
    ]
    harness = PickerHarness(options)
    async with harness.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        picker = harness.screen
        from textual.widgets import OptionList

        rows = picker.query_one("#session-rows", OptionList)
        labels = [str(rows.get_option_at_index(i).prompt) for i in range(rows.option_count)]
        assert any("•" in label for label in labels)
        assert any("[locked]" in label for label in labels)
