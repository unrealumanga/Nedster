"""Tests for the ProgressLine helper in nedster_agent.cli."""

import io
from types import SimpleNamespace

from nedster_agent.cli import ProgressLine


def _clock(at):
    """Return a monotonic-like clock that is already at *at* seconds."""
    _base = [at]  # mutate through closure
    class FakeClock:
        def __call__(self):
            return _base[0]
    return FakeClock()


class TestProgressLineDisabled:

    def test_no_output_when_disabled(self):
        stream = io.StringIO()
        p = ProgressLine(stream, enabled=False)
        p.on_event(SimpleNamespace(kind="model.usage", data={}))
        p.on_event(SimpleNamespace(kind="tool.started", data={"tool": "read_file"}))
        p.close()
        assert stream.getvalue() == ""

    def test_tool_line_noop_when_disabled(self):
        stream = io.StringIO()
        p = ProgressLine(stream, enabled=False)
        p.tool_line("  read_file -> done (10 ms)")
        assert stream.getvalue() == ""


class TestProgressLineEnabled:

    def test_initial_draw_after_delay(self):
        clk = _clock(0.0)
        stream = io.StringIO()
        # No event yet; nothing drawn until on_event or tool_line
        ProgressLine(stream, enabled=True, clock=clk)
        assert stream.getvalue() == ""

    def test_on_event_draws_progress(self):
        clk = _clock(0.3)  # first event triggers draw (>= 0.25)
        stream = io.StringIO()
        p = ProgressLine(stream, enabled=True, clock=clk)
        p.on_event(SimpleNamespace(kind="model.usage", data={}))
        output = stream.getvalue()
        assert "\r" in output
        assert "step 2" in output
        assert "thinking" in output

    def test_progress_line_contains_tool_name(self):
        clk0 = _clock(0.3)
        stream = io.StringIO()
        p = ProgressLine(stream, enabled=True, clock=clk0)
        p.on_event(SimpleNamespace(kind="tool.started", data={"tool": "read_file"}))
        # second event after throttle window to trigger redraw with tool info
        clk2 = _clock(1.0)
        p._clock = clk2  # replace for subsequent draws
        p.on_event(SimpleNamespace(kind="model.usage", data={}))
        output = stream.getvalue()
        assert "read_file" in output

    def test_close_clears_line(self):
        clk = _clock(0.3)
        stream = io.StringIO()
        p = ProgressLine(stream, enabled=True, clock=clk)
        p.on_event(SimpleNamespace(kind="model.usage", data={}))
        p.close()
        output = stream.getvalue()
        assert "\r\x1b[K" in output  # clear at the end

    def test_close_does_nothing_when_disabled(self):
        stream = io.StringIO()
        p = ProgressLine(stream, enabled=False)
        p.close()
        assert stream.getvalue() == ""


class TestThrottling:

    def test_two_events_within_throttle_window_dont_double_draw(self):
        clk = _clock(0.3)  # first event triggers draw
        stream = io.StringIO()
        p = ProgressLine(stream, enabled=True, clock=clk)
        p.on_event(SimpleNamespace(kind="model.usage", data={}))
        count_after_first = stream.getvalue().count("\r")

        # Second event within same throttle window (0.3 + delta < 0.55)
        p.on_event(SimpleNamespace(kind="model.usage", data={}))
        count_after_second = stream.getvalue().count("\r")

        assert count_after_first == 1
        assert count_after_second == 1


class TestAsciiOnly:

    def test_ascii_uses_dashes(self):
        clk = _clock(0.3)
        stream = io.StringIO()
        p = ProgressLine(stream, enabled=True, ascii_only=True, clock=clk)
        p.on_event(SimpleNamespace(kind="model.usage", data={}))
        output = stream.getvalue()
        assert " - " in output
        assert " · " not in output

    def test_non_ascii_uses_dots(self):
        clk = _clock(0.3)
        stream = io.StringIO()
        p = ProgressLine(stream, enabled=True, ascii_only=False, clock=clk)
        p.on_event(SimpleNamespace(kind="model.usage", data={}))
        output = stream.getvalue()
        assert " · " in output
