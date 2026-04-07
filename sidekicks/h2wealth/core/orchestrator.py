"""
H2Wealth - Main Orchestrator
Coordinates: scan → rank → risk → execute → monitor
"""
from __future__ import annotations
import asyncio, logging, time
from typing import List, Optional
from core.config import Config, BotStatus
from core.bybit_client import BybitClient
from core.state_store import StateStore
from core import journal
from signals.engine import SignalEngine
from execution.position_manager import PositionManager

log = logging.getLogger("orchestrator")


class Orchestrator:
    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self.client  = BybitClient(cfg)
        self.state   = StateStore(cfg)
        self.signals_engine: Optional[SignalEngine]  = None
        self.pos_mgr: Optional[PositionManager]       = None
        self._scan_task:    Optional[asyncio.Task]    = None
        self._metrics_task: Optional[asyncio.Task]    = None
        self._symbols:      List[str]                 = []

    async def start(self):
        log.info("H2Wealth starting...")
        journal.init_db()
        await self.client.start()
        await self.state.connect()
        await self.state.set_status(BotStatus.RUNNING)

        # Load instruments
        instruments = await self.client.get_instruments()
        self._symbols = [
            i["symbol"] for i in instruments
            if i.get("quoteCoin") == "USDT"
            and i.get("status") == "Trading"
            and i.get("contractType") == "LinearPerpetual"
        ]
        log.info(f"Found {len(self._symbols)} USDT linear perpetual pairs")

        self.signals_engine = SignalEngine(self.cfg, self.client)
        self.pos_mgr        = PositionManager(self.cfg, self.client, self.state)
        await self.pos_mgr.load_instruments()

        # Start background tasks
        await self.pos_mgr.start_monitor()
        self._scan_task    = asyncio.create_task(self._scan_loop())
        self._metrics_task = asyncio.create_task(self._metrics_loop())
        await self.state.push_log("INFO", f"Bot started. Watching {len(self._symbols)} pairs.")
        log.info("H2Wealth fully started")

    async def stop(self):
        log.info("H2Wealth stopping...")
        await self.state.set_status(BotStatus.STOPPED)
        if self._scan_task:
            self._scan_task.cancel()
        if self._metrics_task:
            self._metrics_task.cancel()
        if self.pos_mgr:
            await self.pos_mgr.stop_monitor()
        await self.client.stop()
        await self.state.disconnect()
        log.info("H2Wealth stopped")

    async def pause(self):
        await self.state.set_status(BotStatus.PAUSED)
        await self.state.push_log("INFO", "Bot paused — no new positions will be opened")

    async def resume(self):
        await self.state.set_status(BotStatus.RUNNING)
        await self.state.push_log("INFO", "Bot resumed")

    async def force_scan(self):
        """Trigger an immediate scan outside the normal loop."""
        log.info("Force scan triggered")
        await self.state.push_log("INFO", "Force scan triggered")
        await self._run_scan()

    async def close_all(self):
        if self.pos_mgr:
            await self.pos_mgr.close_all_positions("manual_close_all")

    # ── Scan Loop ─────────────────────────────────────────────────────────────

    async def _scan_loop(self):
        while True:
            try:
                status = await self.state.get_status()
                if status == BotStatus.RUNNING:
                    await self._run_scan()
                elif status == BotStatus.STOPPED:
                    break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception(f"Scan loop error: {e}")
                await self.state.push_log("ERROR", f"Scan error: {e}")
            await asyncio.sleep(self.cfg.scan_interval_sec)

    async def _run_scan(self):
        now = time.time()
        log.info("Starting pair scan...")
        await self.state.push_log("INFO", "Scanning all pairs...")

        # Expire old signals
        existing = await self.state.get_signals()
        for sig in existing:
            if sig.is_expired(now):
                await self.state.remove_signal(sig.signal_id)

        # Scan
        signals = await self.signals_engine.scan_all(self._symbols)

        # Save to state
        await self.state.clear_signals()
        for sig in signals:
            await self.state.save_signal(sig)
            journal.record_signal(sig)

        await self.state.push_log("INFO",
            f"Scan complete: {len(signals)} signals found. Top: "
            + (f"{signals[0].symbol} {signals[0].side.value} score={signals[0].score:.1f}" if signals else "none")
        )

        # Open positions for top signals
        open_positions  = await self.state.get_open_positions()
        open_symbols    = {p.symbol for p in open_positions}
        equity          = await self.client.get_wallet_balance("USDT")

        for sig in signals:
            if sig.symbol in open_symbols:
                continue
            if len(open_positions) >= self.cfg.max_concurrent:
                break
            pos = await self.pos_mgr.open_from_signal(sig)
            if pos:
                open_positions.append(pos)
                open_symbols.add(sig.symbol)

    # ── Metrics Loop ──────────────────────────────────────────────────────────

    async def _metrics_loop(self):
        while True:
            try:
                await self._update_metrics()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.debug(f"Metrics error: {e}")
            await asyncio.sleep(10)

    async def _update_metrics(self):
        equity   = await self.client.get_wallet_balance("USDT")
        open_pos = await self.state.get_open_positions()
        signals  = await self.state.get_signals()
        perf     = journal.get_performance_summary()
        status   = await self.state.get_status()

        total_pnl  = sum(p.pnl_usdt for p in open_pos)

        metrics = {
            "equity":          round(equity, 2),
            "open_positions":  len(open_pos),
            "active_signals":  len(signals),
            "open_pnl":        round(total_pnl, 4),
            "status":          status.value,
            "total_trades":    perf.get("total", 0) or 0,
            "win_trades":      perf.get("wins", 0) or 0,
            "total_pnl_closed": round(perf.get("total_pnl", 0) or 0, 4),
            "best_trade":      round(perf.get("best_trade", 0) or 0, 4),
            "worst_trade":     round(perf.get("worst_trade", 0) or 0, 4),
            "scan_interval":   self.cfg.scan_interval_sec,
            "pairs_watching":  len(self._symbols),
        }
        await self.state.update_metrics(metrics)
        journal.snapshot_performance(equity, len(open_pos))
