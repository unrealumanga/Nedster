from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import Any

import psutil
from rich.markup import escape
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, Label, ProgressBar, RichLog, Static

from .config import IroncladConfig
from .orchestrator import MythosOrchestrator


class NedsterIroncladApp(App[None]):
    TITLE = "NEDSTER // IRONCLAD"
    SUB_TITLE = "MYTHOS LOCAL AGENT RUNTIME"

    CSS = """
    Screen {
        background: #070A12;
        color: #D9E7FF;
        layout: vertical;
    }
    Header {
        background: #0A1020;
        color: #72F1B8;
        text-style: bold;
        border-bottom: solid #1C315E;
    }
    #hero {
        height: 5;
        padding: 0 2;
        background: #0A1020;
        color: #72F1B8;
        border-bottom: heavy #7C5CFC;
        text-style: bold;
        content-align: left middle;
    }
    #main-grid {
        height: 1fr;
        padding: 1;
    }
    .column {
        height: 1fr;
        margin: 0 1;
        background: #0B1222;
        border: round #1C315E;
    }
    #left { width: 29%; min-width: 30; }
    #center { width: 1fr; }
    #right { width: 30%; min-width: 32; }
    .section-title {
        height: 3;
        padding: 1 2 0 2;
        color: #9EFBD3;
        text-style: bold;
    }
    DataTable {
        height: 1fr;
        background: #0B1222;
        color: #CAD8F2;
        scrollbar-color: #7C5CFC;
        scrollbar-background: #0B1222;
    }
    DataTable > .datatable--header {
        background: #111B32;
        color: #72F1B8;
        text-style: bold;
    }
    RichLog {
        height: 1fr;
        padding: 0 1;
        background: #080D18;
        color: #D9E7FF;
        scrollbar-color: #7C5CFC;
        border-top: solid #172542;
        border-bottom: solid #172542;
    }
    #telemetry {
        height: 8;
        padding: 1 2;
        color: #91A7CA;
    }
    ProgressBar {
        margin: 0 2;
        height: 2;
    }
    ProgressBar > .bar--bar { color: #7C5CFC; }
    ProgressBar > .bar--complete { color: #72F1B8; }
    #command {
        dock: bottom;
        height: 3;
        margin: 0 1 1 1;
        padding: 0 1;
        background: #0E1730;
        color: #FFFFFF;
        border: tall #7C5CFC;
    }
    #command:focus { border: tall #72F1B8; }
    #status-line {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: #111B32;
        color: #8EA6CC;
    }
    Footer {
        background: #080D18;
        color: #8199C2;
    }
    """

    BINDINGS = [
        ("ctrl+c", "cancel_or_quit", "Cancel / Quit"),
        ("ctrl+l", "clear_log", "Clear log"),
        ("ctrl+d", "toggle_dark", "Contrast"),
    ]

    def __init__(self, config: IroncladConfig):
        super().__init__()
        self.config = config
        self.runtime: MythosOrchestrator | None = None
        self.busy = False
        self._last_event = "boot"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(
            "[b #72F1B8]███ NEDSTER[/]  [#7C5CFC]IRONCLAD SWARM[/]\n"
            "[#91A7CA]Router pinned · workers paged · capabilities sealed · state hash-chained[/]",
            id="hero",
            markup=True,
        )
        with Horizontal(id="main-grid"):
            with Vertical(classes="column", id="left"):
                yield Label("MISSION GRAPH", classes="section-title")
                yield DataTable(id="tasks", cursor_type="row", zebra_stripes=True)
            with Vertical(classes="column", id="center"):
                yield Label("LIVE COGNITION + EVENT STREAM", classes="section-title")
                yield RichLog(id="activity", highlight=True, markup=True, wrap=True, auto_scroll=True)
                yield Input(placeholder="Describe the outcome…  (/help for commands)", id="command")
            with Vertical(classes="column", id="right"):
                yield Label("MODEL FLEET", classes="section-title")
                yield DataTable(id="fleet", cursor_type="row", zebra_stripes=True)
                yield Label("SILICON TELEMETRY", classes="section-title")
                yield Static("Probing hardware…", id="telemetry", markup=True)
                yield ProgressBar(total=100, show_eta=False, id="memory")
        yield Static("SAFE-ZONE // initializing", id="status-line")
        yield Footer()

    async def on_mount(self) -> None:
        tasks = self.query_one("#tasks", DataTable)
        tasks.add_column("ID", key="ID")
        tasks.add_column("STEP", key="STEP")
        tasks.add_column("STATE", key="STATE")
        fleet = self.query_one("#fleet", DataTable)
        fleet.add_column("ROLE", key="ROLE")
        fleet.add_column("MODEL", key="MODEL")
        fleet.add_column("BACKEND", key="BACKEND")
        self.query_one("#activity", RichLog).write(
            "[#72F1B8]◆[/] Booting the Ironclad safe-zone…"
        )
        self.set_interval(1.0, self._refresh_telemetry)
        self.run_worker(self._initialize_runtime(), group="startup", exclusive=True)

    async def _initialize_runtime(self) -> None:
        try:
            self.runtime = MythosOrchestrator(self.config, on_event=self._handle_event)
            fleet = await self.runtime.initialize()
            table = self.query_one("#fleet", DataTable)
            for role, model, provider in fleet.as_rows():
                table.add_row(role.upper(), model, provider)
            self.query_one("#status-line", Static).update(
                f"SAFE-ZONE // {self.runtime.profile.name} // sandbox={self.runtime.tools.sandbox.backend} // ready"
            )
            self.query_one("#activity", RichLog).write(
                "[#72F1B8]✓[/] Runtime ready. Local execution is policy-gated."
            )
        except Exception as exc:
            self.query_one("#status-line", Static).update("SAFE-ZONE // degraded // runtime unavailable")
            self.query_one("#activity", RichLog).write(
                f"[#FF6B6B]✕ Initialization failed:[/] {escape(str(exc))}\n"
                "[#91A7CA]Start Ollama and install at least one chat/tool model, then restart.[/]"
            )

    def _refresh_telemetry(self) -> None:
        memory = psutil.virtual_memory()
        self.query_one("#memory", ProgressBar).update(progress=memory.percent)
        if self.runtime:
            hardware = self.runtime.hardware
            text = (
                f"[#72F1B8]{escape(hardware.accelerator)}[/]\n"
                f"RAM  {hardware.ram_available_gb:.1f} / {hardware.ram_total_gb:.1f} GB free\n"
                f"VRAM {hardware.vram_free_gb:.1f} / {hardware.vram_total_gb:.1f} GB free\n"
                f"CPU  {psutil.cpu_percent():.0f}%   MEM {memory.percent:.0f}%\n"
                f"[#7C5CFC]event[/] {escape(self._last_event)}"
            )
            self.query_one("#telemetry", Static).update(text)

    def _handle_event(self, event: dict[str, Any]) -> None:
        self._last_event = str(event.get("kind", "event"))
        kind = self._last_event
        actor = escape(str(event.get("actor", "runtime")))
        log = self.query_one("#activity", RichLog)
        palette = {
            "tool.failed": "#FF6B6B",
            "tool.denied": "#FFB86C",
            "tool.completed": "#72F1B8",
            "step.finished": "#9EFBD3",
            "worker.turn": "#B5A8FF",
        }
        color = palette.get(kind, "#8EA6CC")
        detail = event.get("message") or event.get("reason") or event.get("tool") or ""
        log.write(f"[{color}]◆ {escape(kind)}[/] [#63799E]{actor}[/] {escape(str(detail))}")
        if kind == "plan.created":
            self._render_plan(event.get("plan") or {})
        elif kind == "step.finished":
            self._update_step(event.get("step") or {})

    def _render_plan(self, plan: dict[str, Any]) -> None:
        table = self.query_one("#tasks", DataTable)
        table.clear(columns=False)
        for step in plan.get("steps") or []:
            table.add_row(str(step.get("id")), str(step.get("title")), "PENDING", key=str(step.get("id")))

    def _update_step(self, step: dict[str, Any]) -> None:
        key = str(step.get("id", ""))
        with suppress(Exception):
            self.query_one("#tasks", DataTable).update_cell(key, "STATE", str(step.get("status", "unknown")).upper())

    @on(Input.Submitted, "#command")
    async def submit_objective(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        event.input.value = ""
        if not value:
            return
        if value.startswith("/"):
            await self._command(value)
            return
        if self.busy:
            self.query_one("#activity", RichLog).write("[#FFB86C]A mission is already running.[/]")
            return
        if not self.runtime:
            self.query_one("#activity", RichLog).write("[#FF6B6B]Runtime is not ready.[/]")
            return
        self.run_worker(self._execute(value), group="mission", exclusive=True)

    async def _execute(self, objective: str) -> None:
        assert self.runtime is not None
        self.busy = True
        self.query_one("#status-line", Static).update("SAFE-ZONE // mission active // runtime owns all side effects")
        self.query_one("#activity", RichLog).write(f"\n[b #FFFFFF]MISSION[/] {escape(objective)}")
        try:
            result = await self.runtime.run(objective)
            color = "#72F1B8" if result["passed"] else "#FFB86C"
            self.query_one("#activity", RichLog).write(
                f"[{color}]◆ Mission {'verified' if result['passed'] else 'needs review'}[/] "
                f"{result['completed_steps']}/{result['total_steps']} steps"
            )
        except asyncio.CancelledError:
            self.query_one("#activity", RichLog).write("[#FFB86C]Mission cancelled.[/]")
            raise
        except Exception as exc:
            self.query_one("#activity", RichLog).write(f"[#FF6B6B]Mission failed:[/] {escape(str(exc))}")
        finally:
            self.busy = False
            self.query_one("#status-line", Static).update("SAFE-ZONE // idle // awaiting objective")

    async def _command(self, value: str) -> None:
        command = value.lower().split()[0]
        log = self.query_one("#activity", RichLog)
        if command in {"/quit", "/exit"}:
            self.exit()
        elif command == "/clear":
            log.clear()
        elif command == "/ledger" and self.runtime:
            valid, count, digest = self.runtime.ledger.verify()
            log.write(f"[#72F1B8]Ledger valid={valid} events={count} head={digest[:16]}…[/]")
        elif command == "/fleet" and self.runtime and self.runtime.fleet:
            log.write(json.dumps(self.runtime.fleet.as_rows(), indent=2))
        else:
            log.write("[#91A7CA]/ledger  /fleet  /clear  /quit[/]")

    def action_clear_log(self) -> None:
        self.query_one("#activity", RichLog).clear()

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark

    def action_cancel_or_quit(self) -> None:
        if self.busy:
            self.workers.cancel_group(self, "mission")
        else:
            self.exit()

    async def on_unmount(self) -> None:
        if self.runtime:
            self.runtime.close()


def run_tui(config: IroncladConfig) -> None:
    NedsterIroncladApp(config).run()
