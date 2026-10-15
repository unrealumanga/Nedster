from __future__ import annotations

import pytest

from nedster_agent.widgets.composer import Composer, History
from nedster_agent.widgets.transcript import AssistantMessage, OutcomeBadge, ToolRow

from .helpers import Harness, ready, screen_text, settle, transcript, type_text, wait_for


async def test_text_submitted_while_busy_is_queued_and_sent_after_the_turn(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        await type_text(pilot, "first task")
        await pilot.press("enter")
        await settle(pilot)
        assert h.app.busy and h.fake.submitted == ["first task"]

        await type_text(pilot, "follow up")
        await pilot.press("enter")
        await settle(pilot)
        assert h.fake.submitted == ["first task"]
        assert "queued: follow up" in transcript(h.app)
        assert h.app.queue == ["follow up"]
        assert "1 queued" in screen_text(h.app)

        h.fake.finish("first answer")
        await settle(pilot, 5)
        assert h.fake.submitted == ["first task", "follow up"]
        assert "queued: follow up" not in transcript(h.app)
        assert h.app.queue == []
        h.fake.finish("second answer")
        await settle(pilot, 5)
        assert not h.app.busy
        text = transcript(h.app)
        assert "first answer" in text and "second answer" in text


async def test_cancel_returns_queued_text_to_the_composer(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("one")
        await settle(pilot)
        h.app.submit_text("two")
        await pilot.press("escape")
        await settle(pilot)
        assert h.app.query_one("#prompt", Composer).text == "two"
        assert h.app.queue == []


@pytest.mark.parametrize(
    "data, reason",
    [
        ({"status": "failed", "summary": "exit 1"}, "failed"),
        ({"status": "timeout", "summary": "after 120s"}, "timeout"),
        ({"status": "invalid", "summary": "path outside project"}, "invalid"),
        ({"status": "denied", "summary": "policy"}, "denied"),
        ({"status": "ok", "summary": "ran", "exit_code": 2}, "exit 2"),
        ({"status": "ok", "summary": "0 tests", "tests_run": 0}, "no tests ran"),
    ],
)
async def test_failed_tool_renders_cross_with_reason_never_check(tmp_path, data, reason):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("run tests")
        await settle(pilot)
        h.fake.event("tool.started", call_id="c1", tool="run_tests", arguments={"argv": ["pytest", "-q"]})
        h.fake.event("tool.finished", call_id="c1", tool="run_tests", elapsed_ms=1500, mutated_paths=[], **data)
        await settle(pilot)
        row = h.app.query_one(ToolRow)
        line = row.summary_line().plain
        assert "✗" in line and "✓" not in line
        assert reason in line
        assert any(span.style and "f85149" in str(span.style) for span in row.summary_line().spans)


async def test_successful_tool_renders_check(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("read")
        await settle(pilot)
        h.fake.event("tool.started", call_id="c1", tool="read_file", arguments={"path": "src/main.py"})
        h.fake.event("tool.finished", call_id="c1", tool="read_file", status="ok", summary="42 lines", elapsed_ms=12)
        await settle(pilot)
        line = h.app.query_one(ToolRow).summary_line().plain
        assert "✓ read_file" in line and "src/main.py" in line and "12ms" in line and "42 lines" in line


async def test_streaming_deltas_render_live_then_markdown(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("explain")
        await settle(pilot)
        h.fake.event("model.delta", text="considering the", kind="thinking")
        h.fake.event("model.delta", text="Hel", kind="content")
        h.fake.event("model.delta", text="lo **wor", kind="content")
        await settle(pilot)
        reply = h.app.query_one(AssistantMessage)
        assert reply.content == "Hello **wor" and not reply.final
        screen = screen_text(h.app)
        assert "Hello **wor" in screen
        assert "thinking (15 chars)" in screen  # collapsed
        h.fake.event("model.delta", text="ld**", kind="content")
        h.fake.finish("Hello **world**", "answered")
        await settle(pilot, 5)
        replies = list(h.app.query(AssistantMessage))
        assert len(replies) == 1 and replies[0].final
        screen = screen_text(h.app)
        assert "Hello world" in screen and "**" not in screen
        assert "[ANSWERED]" in screen


async def test_tool_row_expands_to_argv_output_and_diff(tmp_path):
    h = Harness(tmp_path)
    diff = "--- a/src/main.py\n+++ b/src/main.py\n@@ -1 +1 @@\n-print('old')\n+print('new')\n"
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("edit")
        await settle(pilot)
        h.fake.event("tool.started", call_id="c1", tool="edit_file", arguments={"path": "src/main.py"})
        h.fake.event("tool.finished", call_id="c1", tool="edit_file", status="ok", summary="1 replacement",
                     mutated_paths=["src/main.py"], elapsed_ms=5, diff=diff)
        await settle(pilot)
        row = h.app.query_one(ToolRow)
        assert "+print('new')" not in screen_text(h.app)
        await pilot.click(ToolRow)
        await settle(pilot)
        assert row.expanded
        screen = screen_text(h.app)
        assert "path: src/main.py" in screen
        assert "+print('new')" in screen and "-print('old')" in screen
        row.focus()
        await pilot.press("enter")
        await settle(pilot)
        assert not row.expanded
        assert h.app.changed_paths == ["src/main.py"]


async def test_tool_row_loads_diff_for_new_file_outside_git(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        (h.project / "new.py").write_text("x = 1\n", encoding="utf-8")
        h.app.submit_text("create")
        await settle(pilot)
        h.fake.event("tool.started", call_id="c1", tool="write_file", arguments={"path": "new.py"})
        h.fake.event("tool.finished", call_id="c1", tool="write_file", status="ok", summary="written",
                     mutated_paths=["new.py"], elapsed_ms=3)
        await settle(pilot)
        row = h.app.query_one(ToolRow)
        row.action_toggle()
        await wait_for(pilot, lambda: "+x = 1" in screen_text(h.app))


async def test_status_bar_warns_about_host_execution_and_header_is_truthful(tmp_path):
    h = Harness(tmp_path, info={"sandbox": "host", "isolation": "none", "provider": "REMOTE"})
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        screen = screen_text(h.app)
        assert "host — NO isolation" in screen
        assert "REMOTE" in str(h.app.query_one("#identity").render())
        assert "LOCAL" not in str(h.app.query_one("#identity").render())
        assert "ctx" in screen and "idle" in screen


async def test_status_bar_shows_sandbox_and_context(tmp_path):
    h = Harness(tmp_path, info={"sandbox": "bwrap", "isolation": "full"})
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        screen = screen_text(h.app)
        assert "bwrap full" in screen and "1.2k/32.8k" in screen and "nedster-worker" in screen
        assert "LOCAL" in str(h.app.query_one("#identity").render())


async def test_escape_cancels_and_cancelling_persists_until_confirmed(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        await pilot.press("escape")  # idle: nothing happens, app keeps running
        await settle(pilot)
        assert h.app.is_running and h.fake.cancel_requests == 0

        h.app.submit_text("long task")
        await settle(pilot)
        await pilot.press("escape")
        await settle(pilot)
        assert h.fake.cancel_requests == 1
        assert h.app.status_bar.state == "cancelling…"
        h.fake.event("tool.started", call_id="c9", tool="run_command", arguments={"argv": ["sleep", "5"]})
        h.fake.event("model.delta", text="still going")
        await settle(pilot)
        assert h.app.status_bar.state == "cancelling…"
        assert "cancelling…" in screen_text(h.app)
        await pilot.press("escape")  # second Esc while cancelling does not re-request
        assert h.fake.cancel_requests == 1

        h.fake.confirm_cancel()
        await settle(pilot, 5)
        assert h.app.status_bar.state == "idle"
        assert not h.app.busy
        assert "[CANCELLED]" in screen_text(h.app)
        row = h.app.query_one(ToolRow)
        assert row.status == "failed"  # unfinished tool is not shown as success


async def test_up_down_recall_history_and_history_persists(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        composer = h.app.query_one("#prompt", Composer)
        for text in ("alpha", "beta"):
            await type_text(pilot, text)
            await pilot.press("enter")
            await settle(pilot)
            h.fake.finish("ok")
            await settle(pilot, 4)
        await type_text(pilot, "dra")
        await pilot.press("up")
        assert composer.text == "beta"
        await pilot.press("up")
        assert composer.text == "alpha"
        await pilot.press("down")
        assert composer.text == "beta"
        await pilot.press("down")
        assert composer.text == "dra"
    reloaded = History.for_project(h.config.runtime.state_dir, h.project)
    assert reloaded.entries == ["alpha", "beta"]


async def test_multiline_newline_keys_and_paste_do_not_submit(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        composer = h.app.query_one("#prompt", Composer)
        await type_text(pilot, "line1")
        await pilot.press("ctrl+j")
        await type_text(pilot, "line2")
        await pilot.press("shift+enter")
        await type_text(pilot, "x")
        assert composer.text == "line1\nline2\nx"
        composer.focus()
        from textual.events import Paste

        composer.post_message(Paste("pasted\nwith newline\n"))
        await settle(pilot)
        assert "pasted\nwith newline" in composer.text
        assert h.fake.submitted == []
        await pilot.press("enter")
        await settle(pilot)
        assert h.fake.submitted[0].startswith("line1\nline2\nx")


async def test_at_mention_lists_project_files_and_tab_accepts(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        composer = h.app.query_one("#prompt", Composer)
        await type_text(pilot, "look at @sr")
        await settle(pilot)
        menu = h.app.query_one("#completions")
        assert menu.display
        assert composer.completion is not None
        assert composer.completion.items == ("src/main.py", "src/util.py")
        await pilot.press("down")
        await pilot.press("tab")
        await settle(pilot)
        assert composer.text == "look at @src/util.py "
        assert not menu.display


async def test_at_mention_uses_walk_fallback_outside_git(tmp_path):
    from nedster_agent.widgets.composer import project_files

    root = tmp_path / "plain"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("", encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "junk.py").write_text("", encoding="utf-8")
    files = project_files(root)
    assert "pkg/mod.py" in files
    assert not any(f.startswith(".venv") for f in files)


async def test_slash_completion_and_escape_closes_menu_first(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        composer = h.app.query_one("#prompt", Composer)
        await type_text(pilot, "/re")
        await settle(pilot)
        assert composer.completion is not None and "/resume" in composer.completion.items
        h.app.submit_text("busy")
        await settle(pilot)
        await pilot.press("escape")
        assert composer.completion is None
        assert h.fake.cancel_requests == 0  # first Esc only closed the menu


async def test_commands_clear_new_undo_doctor_status_diff(tmp_path):
    tiers = {
        "inference": {"ready": True, "detail": "worker present"},
        "read_only": {"ready": True, "detail": "ok"},
        "execution": {"ready": False, "detail": "docker down", "remedy": "Start Docker"},
    }
    h = Harness(tmp_path, doctor=lambda _config: tiers)
    async with h.app.run_test(size=(100, 40)) as pilot:
        await ready(pilot, h.app)
        h.app._command("/clear")
        await settle(pilot)
        assert transcript(h.app) == "" and h.fake.new_calls == 0

        h.app._command("/undo")
        await settle(pilot)
        assert "Undo is not available" in transcript(h.app)

        h.app._command("/doctor")
        await wait_for(pilot, lambda: "Start Docker" in transcript(h.app))
        text = transcript(h.app)
        assert "inference-ready" in text and "NOT READY" in text and "Start Docker" in text

        h.app._command("/status")
        await settle(pilot)
        assert "sess-1" in transcript(h.app)

        h.app._command("/diff")
        await wait_for(pilot, lambda: "No files changed" in transcript(h.app))
        assert "No files changed" in transcript(h.app)

        h.app._command("/new")
        await settle(pilot)
        assert h.fake.new_calls == 1 and "New conversation sess-2" in transcript(h.app)

        h.app._command("/sessions")
        await settle(pilot)
        assert "Saved sessions are not available" in transcript(h.app)


async def test_state_changing_command_while_busy_keeps_text(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("work")
        await settle(pilot)
        await type_text(pilot, "/new")
        h.app.query_one("#prompt", Composer).close_completion()
        await pilot.press("enter")
        await settle(pilot)
        assert h.app.query_one("#prompt", Composer).text == "/new"
        assert h.fake.new_calls == 0


async def test_undo_calls_session_undo_when_available(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        calls = []

        async def undo_last() -> str:
            calls.append(1)
            return "restored src/main.py"

        h.fake.undo_last = undo_last  # type: ignore[attr-defined]
        h.app._command("/undo")
        await wait_for(pilot, lambda: "restored src/main.py" in transcript(h.app))
        assert calls == [1] and "restored src/main.py" in transcript(h.app)


async def test_quit_command_and_ctrl_c_twice_exit(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        await pilot.press("ctrl+c")
        await settle(pilot)
        assert h.app.is_running
        await pilot.press("ctrl+c")
        await settle(pilot)
        assert not h.app.is_running
    h2 = Harness(tmp_path)
    async with h2.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h2.app)
        h2.app._command("/quit")
        await settle(pilot)
        assert not h2.app.is_running


async def test_acceptance_badges(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 40)) as pilot:
        await ready(pilot, h.app)
        for outcome, label in (("unvalidated", "[UNVALIDATED]"), ("incomplete", "[INCOMPLETE]"),
                               ("completed", "[COMPLETED]")):
            h.app.submit_text(outcome)
            await settle(pilot)
            h.fake.finish("result", outcome, ("tests were not run after the last edit",))
            await settle(pilot, 4)
            badge = list(h.app.query(OutcomeBadge))[-1]
            assert badge.label == label.strip("[]")
        text = transcript(h.app)
        assert "[UNVALIDATED]" in text and "tests were not run after the last edit" in text


async def test_minimum_size_keeps_composer_status_and_transcript_usable(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(80, 24)) as pilot:
        await ready(pilot, h.app)
        for selector in ("#identity", "#work", "#prompt", "#status", "#message", "Footer"):
            assert h.app.query_one(selector).display
        assert h.app.query_one("#work").size.height >= 10
        assert h.app.query_one("#status").size.height == 1


async def test_startup_announces_loaded_project_instructions(tmp_path):
    info = {"instructions": {"name": "AGENTS.md", "size": 1229, "truncated": False}}
    h = Harness(tmp_path, info=info)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        assert "Loaded AGENTS.md (1.2 KB)" in transcript(h.app)


async def test_slash_init_writes_and_refuses_to_overwrite(tmp_path):
    h = Harness(tmp_path)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app._command("/init")
        await settle(pilot)
        path = h.project / "AGENTS.md"
        assert path.is_file()
        content = path.read_text(encoding="utf-8")
        assert "Project instructions" in content

        h.app._command("/init")
        await settle(pilot)
        assert "already exists" in transcript(h.app)
        assert path.read_text(encoding="utf-8") == content

