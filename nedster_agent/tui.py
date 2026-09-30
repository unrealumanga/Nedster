from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from rich.markdown import Markdown
from rich.markup import escape
from textual import on
from textual.app import App, ComposeResult
from textual.events import Key, Resize
from textual.widgets import Footer, Input, RichLog, Static

from ironclad.config import NedsterConfig
from ironclad.orchestrator import NedsterRuntime


class NedsterApp(App[None]):
    TITLE = "Nedster"
    SUB_TITLE = "local AI agent"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen { background: #0b0d10; color: #e6edf3; layout: vertical; }
    #identity { height: 3; padding: 1 2; background: #0b0d10; color: #f0f6fc; text-style: bold; border-bottom: solid #252a33; }
    #work, #details { height: 1fr; padding: 1 3; background: #0b0d10; color: #e6edf3; scrollbar-color: #394150; scrollbar-color-hover: #596579; }
    #details { display: none; }
    #prompt { height: 3; margin: 0 2; padding: 0 1; background: #11151b; color: #f0f6fc; border: round #394150; }
    #prompt:focus { background: #11151b; color: #f0f6fc; border: round #7aa2f7; text-style: none; }
    #prompt > .input--cursor { background: #7aa2f7; color: #0b0d10; text-style: none; }
    #prompt > .input--placeholder, #prompt > .input--suggestion { color: #788290; text-style: none; }
    #message { height: 1; padding: 0 3; background: #0b0d10; color: #8b949e; }
    Footer { background: #0b0d10; color: #8b949e; }
    #too-small { display: none; height: 1fr; content-align: center middle; text-align: center; }
    .no-color #identity, .no-color #work, .no-color #details, .no-color #message, .no-color Footer { color: ansi_default; background: ansi_default; }
    .no-color #prompt { color: ansi_default; background: ansi_default; border: round ansi_default; }
    .no-color #prompt:focus { border: round ansi_default; text-style: none; }
    .ascii #identity { border-bottom: none; }
    .ascii #prompt { border: round ansi_blue; }
    .ascii #prompt:focus { border: round ansi_bright_blue; text-style: none; }
    """

    BINDINGS = [
        ("question_mark", "help", "Help"),
        ("f2", "details", "Details"),
        ("ctrl+l", "clear", "Clear"),
        ("escape", "cancel", "Cancel / Exit"),
    ]

    def __init__(self, config: NedsterConfig, *, ascii_only: bool = False, no_color: bool = False):
        super().__init__()
        self.config = config
        self.ascii_only = ascii_only
        self.no_color_mode = no_color
        self.runtime: NedsterRuntime | None = None
        self.busy = False
        self.last_event = "starting"
        self.details_visible = False
        self.event_buffer: list[str] = []
        self.step_titles: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Static("terminal too small - need at least 80x24", id="too-small")
        yield Static("Nedster  /  project                         LOCAL  ·  starting", id="identity")
        yield RichLog(id="work", markup=True, wrap=True, auto_scroll=True)
        yield RichLog(id="details", markup=True, wrap=False, auto_scroll=True)
        yield Input(
            placeholder="Ask Nedster to build, inspect, or explain…  / for commands",
            id="prompt",
            select_on_focus=False,
        )
        yield Static("● Starting local runtime…", id="message")
        yield Footer(show_command_palette=False)

    async def on_mount(self) -> None:
        if self.no_color_mode:
            self.screen.add_class("no-color")
        if self.ascii_only:
            self.screen.add_class("ascii")
        self._apply_size(self.size.width, self.size.height)
        self.query_one("#prompt", Input).cursor_blink = False
        root = self.config.runtime.project_root
        self.query_one("#work", RichLog).write(
            "[bold #f0f6fc]Nedster[/]  [#8b949e]local coding agent[/]\n"
            f"[#8b949e]{escape(str(root))}[/]\n\n"
            "Ask a question, or describe the outcome you want. Work and tool activity will appear here."
        )
        self.run_worker(self._initialize(), group="startup", exclusive=True)
        self.query_one("#prompt", Input).focus()

    async def _initialize(self) -> None:
        try:
            self.runtime = NedsterRuntime(self.config, on_event=self._event)
            fleet = await self.runtime.initialize()
            worker = fleet.for_role(__import__("ironclad.types", fromlist=["ModelRole"]).ModelRole.CODER).name
            self.query_one("#identity", Static).update(
                f"Nedster  /  {self.config.runtime.project_root.name}"
                f"                         LOCAL  ·  {worker}"
            )
            self._status("● Ready")
        except Exception as exc:
            self.query_one("#identity", Static).update(
                f"Nedster  /  {self.config.runtime.project_root.name}"
                "                         SETUP NEEDED"
            )
            error_style = "bold red" if not self.no_color_mode else "bold"
            self.query_one("#work", RichLog).write(
                f"\n[{error_style}][FAIL] Could not start the local runtime:[/] {escape(str(exc))}\n"
                "Run [bold]nedster setup[/] to install Ollama and matched models, then reopen Nedster."
            )
            self._status("● Setup needed · /doctor for details")

    @staticmethod
    def _tool_label(tool: str) -> str:
        return tool.replace("_", " ").strip().title() or "Tool"

    @staticmethod
    def _result_summary(result: Any) -> str:
        if not isinstance(result, dict):
            return "complete"
        if "tracked_count" in result or "untracked_count" in result:
            return (
                f"{result.get('tracked_count', 0)} tracked, "
                f"{result.get('untracked_count', 0)} untracked"
            )
        for key, noun in (("tests_run", "tests"), ("total_matches", "matches"), ("count", "items")):
            if key in result:
                return f"{result[key]} {noun}"
        if result.get("path"):
            changed = " · changed" if result.get("changed") else ""
            return f"{result['path']}{changed}"
        if "returncode" in result:
            return f"exit {result['returncode']}"
        return "complete"

    def _work_line(self, value: str) -> None:
        self.query_one("#work", RichLog).write(self._text(value))

    def _progress_event(self, event: dict[str, Any]) -> None:
        kind = str(event.get("kind") or "")
        if kind == "response.started":
            self._work_line("[#8b949e]◇ Thinking[/]")
        elif kind == "response.escalated":
            reason = escape(str(event.get("reason") or "verification required"))
            self._work_line(f"[#d2a8ff]◇ Switching to tools[/]  [#8b949e]{reason}[/]")
        elif kind == "plan.created":
            plan = event.get("plan") or {}
            steps = list(plan.get("steps") or []) if isinstance(plan, dict) else []
            self.step_titles.update(
                {
                    str(step.get("id") or ""): str(step.get("title") or "Work")
                    for step in steps
                    if isinstance(step, dict)
                }
            )
            self._work_line(f"\n[bold #d2a8ff]Plan[/]  [#8b949e]{len(steps)} steps[/]")
            for index, step in enumerate(steps, 1):
                title = escape(str(step.get("title") or step.get("objective") or "Work"))
                self._work_line(f"  [#6e7681]{index:>2}[/]  {title}")
        elif kind == "planner.fallback":
            self._work_line("[#d29922]◇ Using a simplified plan[/]")
        elif kind == "swarm.parallel":
            count = len(event.get("step_ids") or [])
            self._work_line(f"[#7aa2f7]◇ Running {count} read-only steps in parallel[/]")
        elif kind == "step.started":
            step = event.get("step") or {}
            step_id = str(step.get("id") or "")
            title = str(step.get("title") or step.get("objective") or "Working")
            self.step_titles[step_id] = title
            role = escape(str(step.get("role") or event.get("actor") or "worker"))
            self._work_line(f"\n[bold #f0f6fc]◆ {escape(title)}[/]  [#6e7681]{role}[/]")
        elif kind == "worker.turn":
            turn = event.get("turn")
            suffix = f" · turn {turn}" if turn else ""
            self._work_line(f"  [#8b949e]◇ Reasoning{suffix}[/]")
        elif kind == "worker.stalled":
            reason = escape(str(event.get("reason") or "retrying with more evidence"))
            self._work_line(f"  [#d29922]↻ Retrying[/]  [#8b949e]{reason}[/]")
        elif kind == "step.restarted":
            title = self.step_titles.get(str(event.get("step_id") or ""), "Step")
            self._work_line(f"  [#d29922]↻ {escape(title)} · alternate approach[/]")
        elif kind == "tool.authorized":
            tool = self._tool_label(str(event.get("tool") or "tool"))
            self._work_line(f"  [#7aa2f7]→[/] {escape(tool)}")
        elif kind == "tool.completed":
            tool = self._tool_label(str(event.get("actor") or "tool"))
            summary = escape(self._result_summary(event.get("result")))
            self._work_line(f"  [#3fb950]✓[/] {escape(tool)}  [#8b949e]{summary}[/]")
        elif kind in {"tool.failed", "tool.denied"}:
            tool = self._tool_label(str(event.get("tool") or event.get("actor") or "tool"))
            reason = escape(str(event.get("error") or event.get("reason") or "failed"))
            self._work_line(f"  [#f85149]✕[/] {escape(tool)}  [#8b949e]{reason}[/]")
        elif kind == "step.finished":
            step = event.get("step") or {}
            verdict = event.get("verdict") or {}
            title = str(step.get("title") or self.step_titles.get(str(step.get("id") or ""), "Step"))
            passed = bool(verdict.get("passed"))
            symbol, color, state = ("✓", "#3fb950", "verified") if passed else ("!", "#d29922", "needs review")
            self._work_line(f"  [{color}]{symbol} {escape(title)}[/]  [#8b949e]{state}[/]")
        elif kind == "run.blocked":
            reason = escape(str(event.get("reason") or "blocked"))
            self._work_line(f"[#f85149]✕ Run blocked[/]  [#8b949e]{reason}[/]")
        elif kind == "run.finished":
            summary = event.get("summary") or {}
            passed = bool(summary.get("passed"))
            completed = summary.get("completed_steps", 0)
            total = summary.get("total_steps", 0)
            symbol, color, state = ("✓", "#3fb950", "Work complete") if passed else ("!", "#d29922", "Partial result")
            self._work_line(f"\n[{color}]{symbol} {state}[/]  [#8b949e]{completed}/{total} steps[/]")

    def _event(self, event: dict[str, Any]) -> None:
        self.last_event = str(event.get("kind", "working"))
        detail = event.get("message") or event.get("reason") or event.get("tool") or ""
        failed = self.last_event.endswith(("failed", "denied"))
        symbol = "[FAIL]" if failed else ">"
        event_line = f"[dim]{symbol} {escape(self.last_event)}[/] {escape(str(detail))}"
        self.event_buffer.append(event_line)
        if len(self.event_buffer) > 500:
            del self.event_buffer[:100]
        if self.details_visible:
            self.query_one("#details", RichLog).write(event_line)
        self._progress_event(event)
        statuses = {
            "runtime.ready": "● Ready",
            "response.started": "● Thinking… · Esc cancels",
            "response.finished": "● Ready",
            "response.escalated": "● Gathering evidence… · Esc cancels",
            "objective.received": "● Thinking… · Esc cancels",
            "planner.fallback": "● Using a simple plan… · Esc cancels",
            "plan.created": "● Plan ready · working… · Esc cancels",
            "step.started": "● Working… · Esc cancels",
            "worker.turn": "● Reasoning… · Esc cancels",
            "tool.authorized": f"● Using {event.get('tool', 'tool')}… · Esc cancels",
            "tool.completed": f"● Used {event.get('actor', 'tool')} · working… · Esc cancels",
            "step.finished": "● Verifying… · Esc cancels",
            "run.finished": "● Ready",
        }
        if failed:
            self._status("● A tool step failed · continuing… · Esc cancels")
        else:
            self._status(statuses.get(self.last_event, "● Working… · Esc cancels"))

    def _status(self, message: str) -> None:
        self.query_one("#message", Static).update(self._text(message))

    def _text(self, value: str) -> str:
        if not self.ascii_only:
            return value
        replacements = {
            "…": "...",
            "·": "|",
            "—": "-",
            "●": "*",
            "◇": ">",
            "◆": ">",
            "→": "->",
            "✓": "OK",
            "✕": "X",
            "↻": "~",
            "›": ">",
        }
        for symbol, fallback in replacements.items():
            value = value.replace(symbol, fallback)
        return value

    def _apply_size(self, width: int, height: int) -> None:
        too_small = width < 80 or height < 24
        self.query_one("#too-small", Static).display = too_small
        self.query_one("#identity").display = not too_small
        self.query_one("#work").display = not too_small and not self.details_visible
        self.query_one("#details").display = not too_small and self.details_visible
        for selector in ("#prompt", "#message", "Footer"):
            self.query_one(selector).display = not too_small

    def on_resize(self, event: Resize) -> None:
        self._apply_size(event.size.width, event.size.height)

    def on_key(self, event: Key) -> None:
        if event.key == "question_mark" and isinstance(self.focused, Input) and not self.focused.value:
            event.prevent_default()
            event.stop()
            self._command("/help")

    @on(Input.Submitted, "#prompt")
    async def submit(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        event.input.value = ""
        if not value:
            return
        if value.startswith("/"):
            self._command(value)
            return
        if self.busy:
            self._status("A task is running · Esc cancels")
            return
        if not self.runtime:
            self._status("Runtime unavailable · run nedster setup")
            return
        self.run_worker(self._execute(value), group="task", exclusive=True)

    async def _execute(self, objective: str) -> None:
        assert self.runtime is not None
        self.busy = True
        log = self.query_one("#work", RichLog)
        self.step_titles.clear()
        log.write(f"\n[bold #f0f6fc]›[/]  [bold #f0f6fc]{escape(objective)}[/]")
        self._status("● Working… · Esc cancels")
        try:
            result = await self.runtime.respond(objective)
            answer = str(result.get("answer") or "I could not produce an answer.").strip()
            log.write("\n[bold #7aa2f7]Nedster[/]")
            log.write(Markdown(answer, code_theme="monokai"))
            grounded = bool(result.get("grounded"))
            passed = bool(result.get("passed"))
            completed = int(result.get("completed_steps") or 0)
            total = int(result.get("total_steps") or 0)
            state = "verified" if passed and grounded else ("complete" if passed else "needs review")
            scope = f" · {completed}/{total} steps" if total else ""
            log.write(f"[#6e7681]{state}{scope}[/]")
        except asyncio.CancelledError:
            log.write("[bold][CANCELLED][/] Task cancelled.")
            raise
        except Exception as exc:
            error_style = "bold red" if not self.no_color_mode else "bold"
            log.write(f"[{error_style}][FAIL] Task failed:[/] {escape(str(exc))}")
        finally:
            self.busy = False
            self._status("● Ready")

    def _command(self, value: str) -> None:
        command = value.lower().split()[0]
        log = self.query_one("#work", RichLog)
        if command in {"/quit", "/exit"}:
            self.exit()
        elif command == "/clear":
            log.clear()
        elif command == "/help":
            log.write("[bold #d2a8ff]Commands[/]  /models  /doctor  /permissions  /clear  /quit")
            log.write("[#8b949e]Keys[/]  ? help  F2 raw events  Ctrl+L clear  Esc cancel/exit")
        elif command == "/models":
            rows = self.runtime.fleet.as_rows() if self.runtime and self.runtime.fleet else []
            log.write(
                "\n".join(f"{role}: {model} ({backend})" for role, model, backend in rows)
                or "No model fleet is active. Run `nedster setup`."
            )
        elif command == "/doctor":
            log.write("Run `nedster doctor` outside the TUI for a complete machine-readable check.")
        elif command == "/permissions":
            policy = self.config.policy
            log.write(f"Autonomous: {policy.autonomous} · automatic risk ceiling: {policy.max_auto_risk}")
        else:
            log.write("Unknown command. Use /help.")

    def action_help(self) -> None:
        self._command("/help")

    def action_details(self) -> None:
        self.details_visible = not self.details_visible
        self.query_one("#work", RichLog).display = not self.details_visible
        details = self.query_one("#details", RichLog)
        details.display = self.details_visible
        if self.details_visible:
            details.clear()
            for line in self.event_buffer:
                details.write(line)
        self._status("● Raw events · F2 returns to conversation" if self.details_visible else "● Ready")

    def action_clear(self) -> None:
        self.query_one("#work", RichLog).clear()

    def action_cancel(self) -> None:
        if self.busy:
            self.workers.cancel_group(self, "task")
        else:
            self.exit()

    async def on_unmount(self) -> None:
        if self.runtime:
            with suppress(Exception):
                self.runtime.close()


def run_tui(config: NedsterConfig, *, ascii_only: bool = False, no_color: bool = False) -> None:
    NedsterApp(config, ascii_only=ascii_only, no_color=no_color).run()
