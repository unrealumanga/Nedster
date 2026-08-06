from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from rich.markup import escape
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Footer, Input, RichLog, Static

from ironclad.config import NedsterConfig
from ironclad.orchestrator import NedsterRuntime


class NedsterApp(App[None]):
    TITLE = "Nedster"
    SUB_TITLE = "local AI agent"

    CSS = """
    Screen { background: #0b0f14; color: #e6edf3; layout: vertical; }
    #identity { height: 3; padding: 1 2; background: #111820; color: #79c0ff; text-style: bold; border-bottom: solid #30363d; }
    #work { height: 1fr; padding: 1 2; background: #0b0f14; scrollbar-color: #388bfd; }
    #prompt { height: 3; margin: 0 1; padding: 0 1; background: #111820; border: tall #388bfd; }
    #prompt:focus { border: tall #56d364; }
    #message { height: 1; padding: 0 2; background: #161b22; color: #8b949e; }
    Footer { background: #111820; color: #8b949e; }
    """

    BINDINGS = [
        ("f1", "help", "Help"),
        ("f2", "details", "Details"),
        ("escape", "cancel", "Cancel"),
        ("ctrl+l", "clear", "Clear"),
    ]

    def __init__(self, config: NedsterConfig):
        super().__init__()
        self.config = config
        self.runtime: NedsterRuntime | None = None
        self.busy = False
        self.last_event = "starting"

    def compose(self) -> ComposeResult:
        yield Static("Nedster  |  project  |  ASK  |  model: starting  |  Ollama: checking", id="identity")
        yield RichLog(id="work", markup=True, wrap=True, auto_scroll=True)
        yield Input(placeholder="Ask Nedster to inspect, explain, build, or fix…  (/help)", id="prompt")
        yield Static("Starting local runtime…", id="message")
        yield Footer()

    async def on_mount(self) -> None:
        root = self.config.runtime.project_root
        self.query_one("#work", RichLog).write(
            f"[bold #79c0ff]Nedster[/] is working in [bold]{escape(str(root))}[/]\n"
            "Describe an outcome. Nedster will inspect first and show progress here."
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
            self.query_one("#work", RichLog).write(
                f"\n[bold red]Could not start the local runtime:[/] {escape(str(exc))}\n"
                "Run [bold]nedster setup[/] to install Ollama and matched models, then reopen Nedster."
            )
            self._status("Setup needed · /doctor for details")

    def _event(self, event: dict[str, Any]) -> None:
        self.last_event = str(event.get("kind", "working"))
        detail = event.get("message") or event.get("reason") or event.get("tool") or ""
        symbol = "x" if self.last_event.endswith(("failed", "denied")) else ">"
        self.query_one("#work", RichLog).write(
            f"[#8b949e]{symbol} {escape(self.last_event)}[/] {escape(str(detail))}"
        )
        self._status(self.last_event)

    def _status(self, message: str) -> None:
        self.query_one("#message", Static).update(message)

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
            result = await self.runtime.run(objective)
            state = "complete" if result.get("passed") else "needs review"
            log.write(f"[bold #56d364]Nedster[/]  Task {state}.")
        except asyncio.CancelledError:
            log.write("[#d29922]Task cancelled.[/]")
            raise
        except Exception as exc:
            log.write(f"[bold red]Task failed:[/] {escape(str(exc))}")
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
            log.write("[bold]Commands[/]  /models  /doctor  /permissions  /clear  /quit\n[bold]Keys[/]  F1 help  F2 details  Esc cancel")
        elif command == "/models":
            rows = self.runtime.fleet.as_rows() if self.runtime and self.runtime.fleet else []
            log.write("\n".join(f"{role}: {model} ({backend})" for role, model, backend in rows) or "No model fleet is active. Run `nedster setup`.")
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
        self._command("/models")

    def action_clear(self) -> None:
        self.query_one("#work", RichLog).clear()

    def action_cancel(self) -> None:
        if self.busy:
            self.workers.cancel_group(self, "task")
        else:
            self._status("Nothing to cancel · /quit exits")

    async def on_unmount(self) -> None:
        if self.runtime:
            with suppress(Exception):
                self.runtime.close()


def run_tui(config: NedsterConfig) -> None:
    NedsterApp(config).run()
