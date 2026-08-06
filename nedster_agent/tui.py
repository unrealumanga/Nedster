from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

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
    Screen { background: ansi_default; color: ansi_default; layout: vertical; }
    #identity { height: 3; padding: 1 2; background: ansi_default; color: ansi_bright_blue; text-style: bold; border-bottom: solid ansi_blue; }
    #work, #details { height: 1fr; padding: 1 2; background: ansi_default; color: ansi_default; scrollbar-color: ansi_blue; }
    #details { display: none; }
    #prompt { height: 1; margin: 0 1; padding: 0 1; background: ansi_default; color: ansi_default; border: none; border-left: thick ansi_blue; }
    #prompt:focus { background: ansi_default; color: ansi_default; border: none; border-left: thick ansi_bright_blue; text-style: none; }
    #prompt > .input--cursor { background: ansi_default; color: ansi_default; text-style: underline; }
    #prompt > .input--placeholder, #prompt > .input--suggestion { color: ansi_default; text-style: none; }
    #message { height: 1; padding: 0 2; background: ansi_default; color: ansi_default; }
    Footer { background: ansi_default; color: ansi_default; }
    #too-small { display: none; height: 1fr; content-align: center middle; text-align: center; }
    .no-color #identity, .no-color #work, .no-color #details, .no-color #message, .no-color Footer { color: ansi_default; background: ansi_default; }
    .no-color #prompt { color: ansi_default; background: ansi_default; border: none; border-left: thick ansi_default; }
    .no-color #prompt:focus { border: none; border-left: thick ansi_default; text-style: none; }
    .ascii #identity { border-bottom: none; }
    .ascii #prompt { border: none; border-left: thick ansi_blue; }
    .ascii #prompt:focus { border: none; border-left: thick ansi_bright_blue; text-style: none; }
    """

    BINDINGS = [
        ("question_mark", "help", "Help"),
        ("f2", "details", "Details"),
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

    def compose(self) -> ComposeResult:
        yield Static("terminal too small - need at least 80x24", id="too-small")
        yield Static("Nedster  |  project  |  ASK  |  model: starting  |  Ollama: checking", id="identity")
        yield RichLog(id="work", markup=True, wrap=True, auto_scroll=True)
        yield RichLog(id="details", markup=True, wrap=False, auto_scroll=True)
        yield Input(
            placeholder="> Ask a question or describe a task  (/help)", id="prompt", select_on_focus=False
        )
        yield Static("Starting local runtime…", id="message")
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
            f"[bold]Project[/]  {escape(str(root))}\nAsk a question or describe an outcome."
        )
        self.run_worker(self._initialize(), group="startup", exclusive=True)
        self.query_one("#prompt", Input).focus()

    async def _initialize(self) -> None:
        try:
            self.runtime = NedsterRuntime(self.config, on_event=self._event)
            fleet = await self.runtime.initialize()
            worker = fleet.for_role(__import__("ironclad.types", fromlist=["ModelRole"]).ModelRole.CODER).name
            self.query_one("#identity", Static).update(
                f"Nedster  |  {self.config.runtime.project_root.name}  |  ASK  |  {worker}  |  Ollama: ready"
            )
            self._status("Ready")
        except Exception as exc:
            self.query_one("#identity", Static).update(
                f"Nedster  |  {self.config.runtime.project_root.name}  |  SETUP NEEDED  |  Ollama: unavailable"
            )
            error_style = "bold red" if not self.no_color_mode else "bold"
            self.query_one("#work", RichLog).write(
                f"\n[{error_style}][FAIL] Could not start the local runtime:[/] {escape(str(exc))}\n"
                "Run [bold]nedster setup[/] to install Ollama and matched models, then reopen Nedster."
            )
            self._status("Setup needed · /doctor for details")

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
        statuses = {
            "runtime.ready": "Ready",
            "response.started": "Thinking… · Esc cancels",
            "response.finished": "Ready",
            "objective.received": "Thinking… · Esc cancels",
            "planner.fallback": "Using a simple plan… · Esc cancels",
            "plan.created": "Plan ready · working… · Esc cancels",
            "step.started": "Working… · Esc cancels",
            "worker.turn": "Thinking… · Esc cancels",
            "tool.authorized": f"Using {event.get('tool', 'tool')}… · Esc cancels",
            "tool.completed": f"Used {event.get('actor', 'tool')} · working… · Esc cancels",
            "step.finished": "Verifying… · Esc cancels",
            "run.finished": "Ready",
        }
        if failed:
            self._status("A tool step failed · continuing… · Esc cancels")
        else:
            self._status(statuses.get(self.last_event, "Working… · Esc cancels"))

    def _status(self, message: str) -> None:
        self.query_one("#message", Static).update(self._text(message))

    def _text(self, value: str) -> str:
        if not self.ascii_only:
            return value
        return value.replace("…", "...").replace("·", "|").replace("—", "-")

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
        log.write(f"\n[bold #f0f6fc]You[/]  {escape(objective)}")
        self._status("Working… · Esc cancels")
        try:
            result = await self.runtime.respond(objective)
            answer = str(result.get("answer") or "I could not produce an answer.").strip()
            log.write(f"[bold]Nedster[/]  {escape(answer)}")
        except asyncio.CancelledError:
            log.write("[bold][CANCELLED][/] Task cancelled.")
            raise
        except Exception as exc:
            error_style = "bold red" if not self.no_color_mode else "bold"
            log.write(f"[{error_style}][FAIL] Task failed:[/] {escape(str(exc))}")
        finally:
            self.busy = False
            self._status("Ready")

    def _command(self, value: str) -> None:
        command = value.lower().split()[0]
        log = self.query_one("#work", RichLog)
        if command in {"/quit", "/exit"}:
            self.exit()
        elif command == "/clear":
            log.clear()
        elif command == "/help":
            log.write(
                "[bold]Commands[/]  /models  /doctor  /permissions  /clear  /quit\n[bold]Keys[/]  ? help  F2 details  Esc cancel/exit"
            )
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
        self._status("Details · F2 returns to conversation" if self.details_visible else "Ready")

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
