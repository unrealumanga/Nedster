"""
H2Wealth - Textual TUI
Rich terminal interface with live updates via Redis pub/sub.
"""

from __future__ import annotations
import asyncio, json, time
from datetime import datetime
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Static,
    DataTable,
    RichLog,
    Button,
    Label,
    Rule,
)
from textual.reactive import reactive
from textual.timer import Timer
from rich.text import Text
from rich.table import Table
from core.config import Config, BotStatus
from core.orchestrator import Orchestrator

cfg = Config()


def fmt_pnl(v: float) -> Text:
    s = f"{v:+.4f}"
    return Text(s, style="bold green" if v >= 0 else "bold red")


def fmt_side(s: str) -> Text:
    return Text(s, style="bold green" if s == "Buy" else "bold red")


def ttl_bar(pct: float, width: int = 10) -> Text:
    filled = int(pct / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    style = "green" if pct < 50 else "yellow" if pct < 75 else "red"
    return Text(bar, style=style)


class MetricsPanel(Static):
    metrics: reactive[dict] = reactive({})

    def render(self) -> str:
        m = self.metrics
        eq = m.get("equity", 0)
        op = m.get("open_pnl", 0)
        tp = m.get("total_pnl_closed", 0)
        st = m.get("status", "—")
        pos = m.get("open_positions", 0)
        sig = m.get("active_signals", 0)
        tot = m.get("total_trades", 0)
        win = m.get("win_trades", 0)
        wr = f"{win / tot * 100:.1f}%" if tot > 0 else "—"
        pairs = m.get("pairs_watching", 0)

        status_color = {"running": "green", "paused": "yellow", "stopped": "red"}.get(
            st, "white"
        )

        return (
            f"[bold blue]⟁ H2WEALTH[/bold blue]   "
            f"Status: [{status_color}]{st.upper()}[/{status_color}]   "
            f"Equity: [cyan]${eq:.2f}[/cyan]   "
            f"Open PNL: {'[green]' if op >= 0 else '[red]'}{op:+.4f}[/{'green' if op >= 0 else 'red'}]   "
            f"Closed PNL: {'[green]' if tp >= 0 else '[red]'}{tp:+.4f}[/{'green' if tp >= 0 else 'red'}]   "
            f"Pos: [white]{pos}[/white]   "
            f"Signals: [blue]{sig}[/blue]   "
            f"Trades: {tot}  Win: [green]{wr}[/green]   "
            f"Pairs: [yellow]{pairs}[/yellow]"
        )


class H2WealthTUI(App):
    CSS = """
    Screen { background: #0d0d0d; }
    MetricsPanel { background: #151515; border: solid #2a2a2a; padding: 1 2; height: 3; }
    #controls { height: 3; background: #151515; border: solid #2a2a2a; align: left middle; padding: 0 1; }
    Button { margin: 0 1; min-width: 14; }
    #main-area { height: 1fr; }
    #left-panel { width: 55%; }
    #right-panel { width: 45%; }
    DataTable { background: #151515; height: 1fr; }
    #log-panel { background: #151515; border: solid #2a2a2a; height: 1fr; }
    .section-label { background: #1c1c1c; color: #888; padding: 0 1; height: 1; }
    """

    BINDINGS = [
        ("p", "pause", "Pause"),
        ("r", "resume", "Resume"),
        ("s", "force_scan", "Scan"),
        ("c", "close_all", "Close All"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, orch: Orchestrator):
        super().__init__()
        self.orch = orch
        self._metrics: dict = {}
        self._positions: list = []
        self._signals: list = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield MetricsPanel(id="metrics-panel")
        with Horizontal(id="controls"):
            yield Button("▶ Resume [r]", id="btn-resume", variant="success")
            yield Button("⏸ Pause  [p]", id="btn-pause", variant="warning")
            yield Button("⟳ Scan   [s]", id="btn-scan")
            yield Button("✕ Close All", id="btn-close-all", variant="error")
            yield Button("↺ Restart", id="btn-restart")
        with Horizontal(id="main-area"):
            with Vertical(id="left-panel"):
                yield Static("  OPEN POSITIONS", classes="section-label")
                yield DataTable(id="pos-table", cursor_type="row")
                yield Static("  ACTIVE SIGNALS", classes="section-label")
                yield DataTable(id="sig-table", cursor_type="row")
            with Vertical(id="right-panel"):
                yield Static("  LIVE LOG", classes="section-label")
                yield RichLog(id="log-panel", highlight=True, markup=True)
        yield Footer()

    def on_mount(self):
        # Setup tables
        pos_table = self.query_one("#pos-table", DataTable)
        pos_table.add_columns(
            "Symbol", "Side", "Entry", "PNL", "Status", "TTL", "Expires"
        )

        sig_table = self.query_one("#sig-table", DataTable)
        sig_table.add_columns("Symbol", "Side", "Score", "OFI", "CVD", "Fund", "Reason")

        # Start async updates
        self.set_interval(2, self._refresh_data)
        asyncio.create_task(self._subscribe_logs())

    async def _refresh_data(self):
        try:
            self._metrics = await self.orch.state.get_metrics()
            self._positions = await self.orch.state.get_open_positions()
            self._signals = await self.orch.state.get_signals()
            self._update_ui()
        except Exception as e:
            pass

    def _update_ui(self):
        # Metrics
        mp = self.query_one("#metrics-panel", MetricsPanel)
        mp.metrics = self._metrics

        # Positions table
        pos_table = self.query_one("#pos-table", DataTable)
        pos_table.clear()
        now = time.time()
        for p in self._positions:
            pnl = p.pnl_usdt
            total = p.signal_expires_at - p.opened_at
            ttlp = min((now - p.opened_at) / max(total, 1) * 100, 100)
            exp = datetime.fromtimestamp(p.signal_expires_at).strftime("%H:%M:%S")
            pos_table.add_row(
                p.symbol,
                fmt_side(p.side.value if hasattr(p.side, "value") else p.side),
                f"{p.entry_price:.4f}",
                fmt_pnl(pnl),
                Text(
                    str(p.status.value if hasattr(p.status, "value") else p.status),
                    style="dim",
                ),
                ttl_bar(ttlp),
                exp,
            )

        # Signals table
        sig_table = self.query_one("#sig-table", DataTable)
        sig_table.clear()
        for s in self._signals[:15]:
            sig_table.add_row(
                s.symbol,
                fmt_side(s.side.value if hasattr(s.side, "value") else s.side),
                Text(f"{s.score:.1f}", style="bold blue"),
                f"{s.ofi_score:.2f}",
                f"{s.cvd_score:.2f}",
                f"{s.funding_score:.2f}",
                Text(s.reason[:40] if s.reason else "", style="dim"),
            )

    async def _subscribe_logs(self):
        try:
            pubsub = await self.orch.state.subscribe("log")
            log_widget = self.query_one("#log-panel", RichLog)
            async for msg in pubsub.listen():
                if msg and msg.get("type") == "message":
                    try:
                        data = json.loads(msg["data"])
                        level = data.get("lvl", "INFO")
                        text = data.get("msg", "")
                        now = datetime.now().strftime("%H:%M:%S")
                        color = {"INFO": "white", "ERROR": "red", "WARN": "yellow"}.get(
                            level, "white"
                        )
                        log_widget.write(f"[{color}][{now}] {level}: {text}[/{color}]")
                    except Exception:
                        pass
        except Exception:
            pass

    # ── Buttons ───────────────────────────────────────────────────────────────

    async def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id
        if bid == "btn-resume":
            await self.orch.resume()
        elif bid == "btn-pause":
            await self.orch.pause()
        elif bid == "btn-scan":
            asyncio.create_task(self.orch.force_scan())
        elif bid == "btn-close-all":
            await self.orch.close_all()
        elif bid == "btn-restart":
            await self.orch.stop()
            await self.orch.start()

    async def action_pause(self):
        await self.orch.pause()

    async def action_resume(self):
        await self.orch.resume()

    async def action_force_scan(self):
        asyncio.create_task(self.orch.force_scan())

    async def action_close_all(self):
        await self.orch.close_all()


async def run_tui():
    orch = Orchestrator(cfg)
    await orch.start()
    app = H2WealthTUI(orch)
    try:
        await app.run_async()
    finally:
        await orch.stop()


if __name__ == "__main__":
    asyncio.run(run_tui())
