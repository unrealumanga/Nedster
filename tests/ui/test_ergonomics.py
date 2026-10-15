"""BP2-UX: status metrics, Ctrl+R history search, completion descriptions/quoting, F1 + footer."""

from __future__ import annotations

import time

from textual.widgets import OptionList, Static

from nedster_agent.widgets.composer import Composer

from .helpers import Harness, ready, screen_text, settle, transcript, type_text, wait_for

# --- status line ----------------------------------------------------------------------------


async def test_status_unknown_until_measured_and_reset_clears_it(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        bar = h.app.status_bar
        assert (bar.prompt_tokens, bar.decode_tps, bar.step) == (None, None, None)
        assert "ctx —" in screen_text(h.app) and "step —" in screen_text(h.app)

        bar.note_usage(0, 0, decode_ms=0)  # a zero prompt is not a measurement
        assert bar.prompt_tokens is None

        bar.note_usage(1000, 50, decode_ms=1000)
        assert bar.prompt_tokens == 1000 and bar.decode_tps == 50
        bar.reset_usage()  # /resume and model switches call this
        assert bar.prompt_tokens is None and bar.decode_tps is None
        assert "ctx —" in screen_text(h.app)


async def test_model_switch_resets_context_metrics(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.status_bar.note_usage(5000, 100, decode_ms=1000)
        h.app._apply_model("qwen3:8b")
        await settle(pilot)
        assert h.app.status_bar.prompt_tokens is None
        assert "ctx —" in screen_text(h.app)


async def test_resume_resets_context_metrics(tmp_path):
    from tests.test_tui import _resume_app, _seed_session

    app = _resume_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await ready(pilot, app)
        older = _seed_session(app, title="older work")
        app.status_bar.note_usage(5000, 100, decode_ms=1000)
        app._command(f"/resume {older[:8]}")
        await settle(pilot)
        assert app.status_bar.prompt_tokens is None
        assert "ctx —" in screen_text(app)


# --- Ctrl+R reverse history search ----------------------------------------------------------


async def test_ctrl_r_filters_cycles_and_enter_puts_match_in_composer(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        composer = h.app.query_one("#prompt", Composer)
        for text in ("alpha", "beta", "gamma", "alpha two"):
            composer.prompt_history.add(text)

        await type_text(pilot, "draft")
        await pilot.press("ctrl+r")
        await settle(pilot)
        assert composer.text == "alpha two"  # most recent first
        hint = h.app.query_one("#search-hint", Static)
        assert hint.display and "reverse-i-search" in str(hint.render())

        await type_text(pilot, "al")
        await settle(pilot)
        assert composer.text == "alpha two"
        await pilot.press("ctrl+r")  # cycle to an older match
        assert composer.text == "alpha"
        await pilot.press("ctrl+r")  # already the oldest: unchanged
        assert composer.text == "alpha"

        await pilot.press("enter")  # accepts into the composer; must not send
        await settle(pilot)
        assert composer.text == "alpha"
        assert h.fake.submitted == []
        assert not composer.search_active
        assert not h.app.query_one("#search-hint", Static).display


async def test_ctrl_r_escape_restores_the_draft(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        composer = h.app.query_one("#prompt", Composer)
        composer.prompt_history.add("remembered prompt")

        await type_text(pilot, "keep me")
        await pilot.press("ctrl+r")
        await settle(pilot)
        assert composer.text == "remembered prompt"
        await pilot.press("escape")
        await settle(pilot)
        assert composer.text == "keep me"
        assert not composer.search_active
        assert not h.app.query_one("#search-hint", Static).display
        assert h.fake.submitted == []


# --- slash completion descriptions and argument rejection -----------------------------------


async def test_slash_completion_shows_descriptions(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        composer = h.app.query_one("#prompt", Composer)
        await type_text(pilot, "/res")
        await settle(pilot)
        menu = h.app.query_one("#completions", OptionList)
        assert menu.display
        labels = [str(menu.get_option_at_index(i).prompt) for i in range(menu.option_count)]
        assert any(label.startswith("/resume") and "resume a session" in label for label in labels)
        assert composer.completion is not None and "/resume" in composer.completion.items


async def test_no_arg_command_rejects_extra_text_with_usage(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app._command("/help extra")
        await settle(pilot)
        text = transcript(h.app)
        assert "/help takes no arguments" in text and "usage: /help" in text
        assert "Commands" not in text  # the handler did not run

        h.app._command("/model")  # an argument-taking command is still allowed
        await settle(pilot)


# --- @file inventory refresh + quoted paths --------------------------------------------------


async def test_file_inventory_refreshes_after_a_changing_turn_and_respects_ttl(tmp_path):
    current = ["src/main.py"]
    calls: list[int] = []

    def source():
        calls.append(1)
        return list(current)

    h = Harness(tmp_path, files=source)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        await wait_for(pilot, lambda: h.app._files_cache == ["src/main.py"])

        h.app._files_at = time.monotonic()
        before = len(calls)
        await h.app._refresh_files()  # fresh cache: no rescan
        assert len(calls) == before
        await h.app._refresh_files(force=True)  # forced bypasses the TTL
        assert len(calls) == before + 1

        current.append("src/new.py")
        h.app.submit_text("add a file")
        await settle(pilot)
        h.fake.event("tool.started", call_id="c1", tool="write_file", arguments={"path": "src/new.py"})
        h.fake.event("tool.finished", call_id="c1", tool="write_file", status="ok", summary="written",
                     mutated_paths=["src/new.py"], elapsed_ms=3)
        h.fake.finish("done")
        await wait_for(pilot, lambda: h.app._files_cache == ["src/main.py", "src/new.py"])
        assert h.app.query_one("#prompt", Composer).files() == ["src/main.py", "src/new.py"]


async def test_file_inventory_refreshes_periodically_when_idle(tmp_path, monkeypatch):
    import nedster_agent.tui as tui

    monkeypatch.setattr(tui, "FILE_REFRESH_S", 0.05)
    current = ["a.py"]
    h = Harness(tmp_path, files=lambda: list(current))
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        await wait_for(pilot, lambda: h.app._files_cache == ["a.py"])
        current.append("b.py")
        h.app._files_at = time.monotonic() - 1.0  # pretend the cache aged past the (patched) TTL
        await wait_for(pilot, lambda: h.app._files_cache == ["a.py", "b.py"])


async def test_quoted_at_path_with_spaces_completes_and_inserts_quoted(tmp_path):
    files = ["src/my file.txt", "src/other.py"]
    h = Harness(tmp_path, files=lambda: files)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        composer = h.app.query_one("#prompt", Composer)
        await wait_for(pilot, lambda: composer.files() == ["src/my file.txt", "src/other.py"])

        await type_text(pilot, 'see @"my')
        await settle(pilot)
        menu = h.app.query_one("#completions", OptionList)
        assert menu.display
        assert composer.completion is not None
        assert composer.completion.items == ("src/my file.txt",)
        await pilot.press("tab")
        await settle(pilot)
        assert composer.text == 'see @"src/my file.txt" '


# --- F1 and footer ---------------------------------------------------------------------------


async def test_f1_opens_help_and_footer_lists_the_key_hints(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(120, 30)) as pilot:
        await ready(pilot, h.app)
        screen = screen_text(h.app)
        assert "f1 help" in screen
        assert "f4 models" in screen
        assert "history" in screen
        assert "esc stop" in screen
        await pilot.press("f1")
        await settle(pilot)
        assert "Commands" in transcript(h.app)
