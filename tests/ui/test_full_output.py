"""BP2-VIEW: the full-output modal, Ctrl+O verbose mode and the follow-the-tail hint."""

from __future__ import annotations

from nedster_agent.widgets.output_viewer import OutputViewer
from nedster_agent.widgets.transcript import (
    FULL_OUTPUT_MAX_CHARS,
    ToolRow,
    render_full_output,
    replay_widgets,
)

from .helpers import Harness, ready, screen_text, settle, transcript, wait_for

# --- render_full_output -------------------------------------------------------------------------


def test_render_full_output_is_lossless_under_the_cap():
    assert render_full_output("short") == "short"


def test_render_full_output_truncates_with_a_byte_count():
    text = "x" * (FULL_OUTPUT_MAX_CHARS + 500)
    rendered = render_full_output(text)
    assert rendered.startswith("x" * FULL_OUTPUT_MAX_CHARS)
    assert rendered.endswith(f"[truncated, {len(text.encode('utf-8'))} bytes total]")


# --- the modal ----------------------------------------------------------------------------------


async def test_pressing_o_on_a_tool_row_opens_the_full_output_modal(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("run tests")
        await settle(pilot)
        h.fake.event("tool.started", call_id="c1", tool="run_tests", arguments={"argv": ["pytest", "-q"]})
        h.fake.event("tool.finished", call_id="c1", tool="run_tests", status="ok", summary="ran",
                     output="first line\nsecond line", elapsed_ms=5)
        await settle(pilot)

        row = h.app.query_one(ToolRow)
        row.focus()
        await pilot.press("o")
        await wait_for(pilot, lambda: isinstance(h.app.screen, OutputViewer))
        await wait_for(pilot, lambda: "first line" in screen_text(h.app))

        await pilot.press("escape")
        await settle(pilot)
        assert not isinstance(h.app.screen, OutputViewer)


async def test_output_viewer_loads_lazily_and_labels_historical_output(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        calls: list[int] = []

        def loader() -> str:
            calls.append(1)
            return "FULL STORED OUTPUT"

        h.app.push_screen(OutputViewer(h.app.glyphs, title="run_tests", loader=loader, historical=True))
        await wait_for(pilot, lambda: "FULL STORED OUTPUT" in screen_text(h.app))
        assert calls == [1]
        assert "historical output" in screen_text(h.app).lower()


async def test_output_viewer_says_when_nothing_was_stored(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.push_screen(OutputViewer(h.app.glyphs, title="read_file", loader=lambda: ""))
        await wait_for(pilot, lambda: "no stored output" in screen_text(h.app).lower())


def test_replayed_tool_rows_are_marked_historical_and_carry_a_loader():
    from nedster_agent.widgets.glyphs import Glyphs

    entries = [{"role": "tool", "text": "old output", "meta": {"tool": "run_tests", "call_id": "c9"}}]
    rows = replay_widgets(Glyphs(), entries, output_loader=lambda call_id: (lambda: f"stored:{call_id}"))
    row = rows[0]
    assert isinstance(row, ToolRow)
    assert row.historical is True
    assert row.output_loader is not None and row.output_loader() == "stored:c9"
    assert "not replayed" in row.summary_line().plain


# --- Ctrl+O verbose -----------------------------------------------------------------------------


async def test_ctrl_o_expands_every_tool_row_and_auto_expands_new_ones(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("run")
        await settle(pilot)
        h.fake.event("tool.started", call_id="c1", tool="run_tests", arguments={"argv": ["pytest"]})
        h.fake.event("tool.finished", call_id="c1", tool="run_tests", status="ok", summary="ran", elapsed_ms=5)
        await settle(pilot)
        first = h.app.query_one(ToolRow)
        assert not first.expanded

        await pilot.press("ctrl+o")
        await settle(pilot)
        assert h.app.verbose and first.expanded

        h.fake.event("tool.started", call_id="c2", tool="read_file", arguments={"path": "src/main.py"})
        await settle(pilot)
        second = [r for r in h.app.query(ToolRow) if r.call_id == "c2"][0]
        assert second.expanded  # verbose mode applied to the new row too

        await pilot.press("ctrl+o")
        await settle(pilot)
        assert not h.app.verbose and not first.expanded


# --- follow-the-tail hint -----------------------------------------------------------------------


async def test_new_output_while_scrolled_up_shows_the_follow_hint(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        for i in range(200):
            h.app._line(f"history line {i}")
        await settle(pilot)
        assert h.app.transcript.following

        h.app.transcript.scroll_to(y=0, animate=False)
        await settle(pilot)
        assert not h.app.transcript.following

        h.fake.event("model.delta", text="fresh output", kind="content")
        await settle(pilot)
        assert h.app.transcript.new_output
        assert h.app.query_one("#follow-hint").display
        assert "End to follow" in screen_text(h.app)

        h.app.transcript.scroll_end(animate=False)
        await settle(pilot)
        assert h.app.transcript.following and not h.app.transcript.new_output
        assert not h.app.query_one("#follow-hint").display


async def test_following_keeps_scrolling_to_the_end(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        assert h.app.transcript.following
        for i in range(200):
            h.app._line(f"line {i}")
        await settle(pilot)
        assert h.app.transcript.is_vertical_scroll_end
        assert not h.app.query_one("#follow-hint").display
        assert "line 199" in transcript(h.app)
