from __future__ import annotations

import pytest
from rich.text import Text

from nedster_agent.widgets.glyphs import Glyphs
from nedster_agent.widgets.session import provider_label
from nedster_agent.widgets.transcript import OutcomeBadge, ToolRow

from .helpers import Harness, ready, screen_text, settle


async def _busy_screen(h: Harness, pilot) -> None:
    await ready(pilot, h.app)
    h.app._command("/help")
    h.app.submit_text("do the thing")
    await settle(pilot)
    h.app.submit_text("then this")  # queued
    h.fake.note("◆ Inspect · plan step", "info")
    h.fake.event("model.delta", text="hmm… “quoted”", kind="thinking")
    h.fake.event("model.delta", text="Working — almost • done", kind="content")
    h.fake.event("tool.started", call_id="a", tool="run_tests", arguments={"argv": ["pytest"]})
    h.fake.event("tool.finished", call_id="a", tool="run_tests", status="ok", tests_run=0, summary="0 tests")
    h.fake.event("tool.started", call_id="b", tool="edit_file", arguments={"path": "src/main.py"})
    h.fake.event("tool.finished", call_id="b", tool="edit_file", status="ok", summary="edited",
                 mutated_paths=["src/main.py"], diff="--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n")
    await settle(pilot)
    for row in h.app.query(ToolRow):
        row.action_toggle()
    await pilot.press("escape")  # cancelling…
    await settle(pilot, 4)


async def test_ascii_mode_renders_no_non_ascii_glyphs_anywhere(tmp_path):
    h = Harness(tmp_path, ascii_only=True, info={"sandbox": "host", "isolation": "none"})
    async with h.app.run_test(size=(100, 50)) as pilot:
        await _busy_screen(h, pilot)
        screen = screen_text(h.app)
        assert "cancelling..." in screen and "host - NO isolation" in screen
        assert "X run_tests" in screen
        bad = sorted({ch for ch in screen if ord(ch) > 127})
        assert bad == [], f"non-ASCII glyphs in ASCII mode: {bad}"
        h.fake.confirm_cancel()
        await settle(pilot, 5)
        screen = screen_text(h.app)
        assert "[CANCELLED]" in screen
        assert sorted({ch for ch in screen if ord(ch) > 127}) == []


async def test_no_color_mode_sets_no_colors_on_rendered_rows(tmp_path):
    h = Harness(tmp_path, no_color=True)
    async with h.app.run_test(size=(100, 40)) as pilot:
        await _busy_screen(h, pilot)
        assert h.app.screen.has_class("no-color")
        rows = list(h.app.query(ToolRow))
        renderables: list[Text] = [row.summary_line() for row in rows]
        renderables.append(h.app.status_bar.render())  # type: ignore[arg-type]
        for text in renderables:
            for span in text.spans:
                assert "#" not in str(span.style), f"color leaked: {span.style}"
        assert "✗" in rows[0].summary_line().plain  # glyphs stay, only colors go


def test_glyph_modes():
    plain = Glyphs(ascii_only=True, no_color=True)
    assert plain.icon("ok") == "OK" and plain.icon("fail") == "X"
    assert plain.style("fail") == "bold reverse" and "#" not in plain.style("ok")
    assert plain.text("café → ✓") == "caf? -> OK"
    rich = Glyphs()
    assert rich.icon("fail") == "✗" and "#f85149" in rich.style("fail")
    assert rich.bar(50, 100, 4) == "██░░"


def test_badge_no_color_has_no_color_style():
    badge = OutcomeBadge(Glyphs(no_color=True), "incomplete", ["validation failing"])
    rendered = badge.render()
    first = rendered.renderables[0]  # type: ignore[attr-defined]
    assert first.plain == "[INCOMPLETE]"
    assert all("#" not in str(span.style) for span in first.spans)


@pytest.mark.parametrize(
    "endpoint, model, label",
    [
        ("http://127.0.0.1:11434", "qwen3:8b", "LOCAL"),
        ("http://localhost:11434", "qwen3:8b", "LOCAL"),
        ("http://[::1]:11434", "qwen3:8b", "LOCAL"),
        ("http://10.0.0.5:11434", "qwen3:8b", "REMOTE"),
        ("http://gpu-box.lan:11434", "qwen3:8b", "REMOTE"),
        ("http://127.0.0.1:11434", "gpt-oss:120b-cloud", "CLOUD"),
        ("http://127.0.0.1:11434", "deepseek-v3.1:671b:cloud", "CLOUD"),
    ],
)
def test_provider_label_is_truthful(endpoint, model, label):
    assert provider_label(endpoint, model) == label
