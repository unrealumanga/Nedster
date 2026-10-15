from __future__ import annotations

from pathlib import Path

from textual.widgets import Static

from nedster_agent import personality

from .helpers import Harness, ready, settle, transcript, type_text


def _message(app) -> str:
    return str(app.query_one("#message", Static).render())


async def test_ui_mode_shows_a_thinking_line_and_a_done_line_not_the_answer(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    h = Harness(tmp_path, soul="ui")
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("do a thing")
        await settle(pilot)
        assert _message(h.app) == personality.pick("thinking", h.app.turn_number)

        h.fake.finish("here is the answer", "completed")
        await settle(pilot, 5)
        assert _message(h.app) == personality.pick("done_ok", h.app.turn_number)
        text = transcript(h.app)
        assert "here is the answer" in text
        for moment in ("thinking", "done_ok", "done_answered", "done_unvalidated", "done_failed", "idle"):
            for line in personality.lines(moment):
                assert line not in text


async def test_off_mode_keeps_the_plain_ready_message(tmp_path: Path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        assert _message(h.app) == "Ready"
        h.app.submit_text("work")
        await settle(pilot)
        h.fake.finish("done", "completed")
        await settle(pilot, 5)
        for moment in ("thinking", "done_ok", "done_failed"):
            assert _message(h.app) != personality.pick(moment, h.app.turn_number)


async def test_soul_command_reports_mode_path_and_size(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(cfg))
    h = Harness(tmp_path, soul="full")
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        path = cfg / "soul.md"
        cfg.mkdir(parents=True, exist_ok=True)
        path.write_text("x" * 10, encoding="utf-8")
        h.app._command("/soul")
        await settle(pilot)
        text = transcript(h.app)
        assert f"Soul: full · {path} · 10 bytes" in text


async def test_soul_remember_confirms_then_writes_and_empty_enter_cancels(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(cfg))
    h = Harness(tmp_path, soul="ui")
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app._command("/soul remember Be blunt")
        await settle(pilot)
        assert h.app._pending_soul is not None
        assert "Will add: - Be blunt" in transcript(h.app)

        await type_text(pilot, "y")
        await pilot.press("enter")
        await settle(pilot)
        assert h.app._pending_soul is None
        assert "- Be blunt" in (cfg / "soul.md").read_text(encoding="utf-8")

        h.app._command("/soul remember Another thing")
        await settle(pilot)
        await pilot.press("enter")  # empty = no
        await settle(pilot)
        assert h.app._pending_soul is None
        assert "Another thing" not in (cfg / "soul.md").read_text(encoding="utf-8")


async def test_soul_reset_restores_the_template(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(cfg))
    h = Harness(tmp_path, soul="full")
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        path = cfg / "soul.md"
        cfg.mkdir(parents=True, exist_ok=True)
        path.write_text("# custom\n", encoding="utf-8")
        h.app._command("/soul reset")
        await settle(pilot)
        await type_text(pilot, "yes")
        await pilot.press("enter")
        await settle(pilot)
        from ironclad.engine.prompt import DEFAULT_SOUL_TEMPLATE

        assert path.read_text(encoding="utf-8") == DEFAULT_SOUL_TEMPLATE
