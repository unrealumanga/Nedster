"""
H2Wealth - Position Manager
Opens positions, monitors them, handles TP cascade and expiry closure.
"""

from __future__ import annotations
import asyncio, logging, time, uuid, math
from typing import Dict, List, Optional
from core.config import Config, Position, Signal, Side, PositionStatus, BotStatus
from core.bybit_client import BybitClient
from core.state_store import StateStore
from core import journal
from execution.risk import RiskEngine, TradeParams

log = logging.getLogger("position_mgr")


class PositionManager:
    def __init__(self, cfg: Config, client: BybitClient, state: StateStore):
        self.cfg = cfg
        self.client = client
        self.state = state
        self.risk = RiskEngine(cfg)
        self._instruments: List[Dict] = []
        self._monitor_task: Optional[asyncio.Task] = None

    async def load_instruments(self):
        self._instruments = await self.client.get_instruments()
        log.info(f"Loaded {len(self._instruments)} instruments")

    # ── Opening Positions ─────────────────────────────────────────────────────

    async def open_from_signal(self, signal: Signal) -> Optional[Position]:
        """Attempt to open a position from a signal. Returns Position or None."""
        try:
            # Check bot not paused
            status = await self.state.get_status()
            if status != BotStatus.RUNNING:
                log.info(f"Bot {status.value}, skipping {signal.symbol}")
                return None

            # Check not already in this symbol
            open_pos = await self.state.get_open_positions()
            if any(p.symbol == signal.symbol for p in open_pos):
                log.info(f"Already have position in {signal.symbol}, skipping")
                return None

            # Get equity
            equity = await self.client.get_wallet_balance("USDT")
            if equity <= 0:
                log.warning("Zero equity, cannot open position")
                return None

            # Get instrument params
            qty_step, price_step, min_qty = self.risk.get_instrument_params(
                self._instruments, signal.symbol
            )

            # Compute trade params
            params = self.risk.compute_trade(
                signal,
                equity,
                len(open_pos),
                qty_step=qty_step,
                price_step=price_step,
                min_qty=min_qty,
            )
            if not params:
                return None

            # Set leverage
            await self.client.set_leverage(signal.symbol, params.leverage)
            await asyncio.sleep(0.1)

            # Place market order with SL attached
            order_id = f"h2w_{signal.symbol}_{int(time.time())}"
            result = await self.client.place_order(
                symbol=params.symbol,
                side=params.side,
                qty=params.qty,
                order_type="Market",
                sl=params.sl_price,
                order_link_id=order_id,
            )

            if result.get("retCode", -1) != 0:
                msg = result.get("retMsg", "unknown error")
                log.error(f"Order failed {signal.symbol}: {msg}")
                await self.state.push_log(
                    "ERROR", f"Open failed {signal.symbol}: {msg}"
                )
                return None

            bybit_order_id = result.get("result", {}).get("orderId", "")

            # Set TP levels via trading-stop (cascade)
            await asyncio.sleep(0.3)
            await self.client.set_trading_stop(
                symbol=params.symbol,
                side=params.side,
                tp=params.tp1_price,  # TP1 initially; we manage 2&3 in monitor
            )

            now = time.time()
            pos = Position(
                position_id=f"pos_{signal.symbol}_{int(now)}_{uuid.uuid4().hex[:6]}",
                symbol=params.symbol,
                side=params.side,
                entry_price=params.entry,
                qty=params.qty,
                leverage=params.leverage,
                sl_price=params.sl_price,
                tp1_price=params.tp1_price,
                tp2_price=params.tp2_price,
                tp3_price=params.tp3_price,
                signal_id=signal.signal_id,
                opened_at=now,
                signal_expires_at=signal.expires_at,
                status=PositionStatus.OPEN,
                bybit_order_id=bybit_order_id,
            )

            await self.state.save_position(pos)
            journal.record_trade_open(pos, signal)
            await self.state.push_log(
                "INFO",
                f"Opened {params.side.value} {params.symbol} @ {params.entry:.4f} "
                f"qty={params.qty} SL={params.sl_price:.4f} TP1={params.tp1_price:.4f}",
            )
            log.info(f"Opened {pos.position_id}")
            return pos

        except Exception as e:
            log.exception(f"open_from_signal error: {e}")
            await self.state.push_log("ERROR", f"Open exception {signal.symbol}: {e}")
            return None

    # ── Monitor Loop ──────────────────────────────────────────────────────────

    async def start_monitor(self):
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        log.info("Position monitor started")

    async def stop_monitor(self):
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        while True:
            try:
                await self._check_positions()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception(f"Monitor loop error: {e}")
            await asyncio.sleep(5)  # check every 5 seconds

    async def _check_positions(self):
        open_positions = await self.state.get_open_positions()
        if not open_positions:
            return

        # Get live position data from exchange
        live_pos = await self.client.get_positions()
        live_map = {p["symbol"]: p for p in live_pos if float(p.get("size", "0")) > 0}

        now = time.time()

        for pos in open_positions:
            try:
                live = live_map.get(pos.symbol)

                # ── Position closed by exchange (SL or TP hit) ──
                if not live:
                    await self._handle_closed_externally(pos)
                    continue

                # Live data
                last_price = float(live.get("markPrice", 0))
                unrealised = float(live.get("unrealisedPnl", 0))
                pos.pnl_usdt = unrealised

                # ── TP1 management ──────────────────────────────
                if pos.status == PositionStatus.OPEN:
                    tp1_hit = (
                        pos.side == Side.BUY and last_price >= pos.tp1_price
                    ) or (pos.side == Side.SELL and last_price <= pos.tp1_price)
                    if tp1_hit:
                        await self._handle_tp1_hit(pos, last_price)
                        continue

                # ── TP2 management ──────────────────────────────
                elif pos.status == PositionStatus.TP1_HIT:
                    tp2_hit = (
                        pos.side == Side.BUY and last_price >= pos.tp2_price
                    ) or (pos.side == Side.SELL and last_price <= pos.tp2_price)
                    if tp2_hit:
                        await self._handle_tp2_hit(pos, last_price)
                        continue

                # ── Signal expiry guardian ──────────────────────
                ttl_pct = pos.ttl_pct(now)
                if ttl_pct >= self.cfg.signal_expire_profit_pct:
                    if unrealised > 0:
                        log.info(
                            f"Signal TTL {ttl_pct:.0f}% expired, closing profitable {pos.symbol}"
                        )
                        await self.close_position(pos, reason="signal_expiry_profit")
                        continue
                    elif ttl_pct >= 100:
                        log.info(f"Signal fully expired, closing {pos.symbol}")
                        await self.close_position(pos, reason="signal_expired")
                        continue

                # Save updated PNL
                await self.state.save_position(pos)

            except Exception as e:
                log.error(f"Check position {pos.position_id} error: {e}")

    async def _handle_tp1_hit(self, pos: Position, price: float):
        """TP1 hit: partially close, move SL to breakeven."""
        log.info(f"TP1 hit {pos.symbol} @ {price:.4f}")

        qty_step, _, min_qty = self.risk.get_instrument_params(
            self._instruments, pos.symbol
        )

        # Close TP1_CLOSE_PCT% of position
        raw_close = pos.qty * (self.cfg.tp1_close_pct / 100.0)
        close_qty = (
            max(min_qty, math.floor(raw_close / qty_step) * qty_step)
            if qty_step
            else raw_close
        )
        close_qty = round(close_qty, 6)

        if close_qty > 0:
            await self.client.close_position(pos.symbol, pos.side, close_qty)

        # Move SL to breakeven (entry + fees)
        be_price = self.risk.breakeven_price(pos.entry_price, pos.side)

        # Try to set SL and advance TP. If it fails (due to price spike making TP invalid), just set SL.
        res = await self.client.set_trading_stop(
            symbol=pos.symbol,
            side=pos.side,
            sl=be_price,
            tp=pos.tp2_price,  # advance TP to TP2
        )
        if res.get("retCode", -1) != 0:
            log.warning(
                f"Failed to set SL/TP for {pos.symbol}, trying SL only. Reason: {res.get('retMsg')}"
            )
            await self.client.set_trading_stop(
                symbol=pos.symbol,
                side=pos.side,
                sl=be_price,
            )

        pos.sl_price = be_price
        pos.status = PositionStatus.TP1_HIT
        await self.state.save_position(pos)
        await self.state.push_log(
            "INFO",
            f"TP1 hit {pos.symbol} @ {price:.4f}, SL moved to breakeven {be_price:.4f}, TP→TP2",
        )

    async def _handle_tp2_hit(self, pos: Position, price: float):
        """TP2 hit: partially close, move SL above TP1 (lock in profit)."""
        log.info(f"TP2 hit {pos.symbol} @ {price:.4f}")

        qty_step, _, min_qty = self.risk.get_instrument_params(
            self._instruments, pos.symbol
        )

        # Close TP2_CLOSE_PCT% of position
        raw_close = pos.qty * (self.cfg.tp2_close_pct / 100.0)
        close_qty = (
            max(min_qty, math.floor(raw_close / qty_step) * qty_step)
            if qty_step
            else raw_close
        )
        close_qty = round(close_qty, 6)

        if close_qty > 0:
            await self.client.close_position(pos.symbol, pos.side, close_qty)

        # Move SL to TP1 level (guaranteed profit)
        res = await self.client.set_trading_stop(
            symbol=pos.symbol,
            side=pos.side,
            sl=pos.tp1_price,
            tp=pos.tp3_price,
        )
        if res.get("retCode", -1) != 0:
            log.warning(
                f"Failed to set SL/TP for {pos.symbol}, trying SL only. Reason: {res.get('retMsg')}"
            )
            await self.client.set_trading_stop(
                symbol=pos.symbol,
                side=pos.side,
                sl=pos.tp1_price,
            )

        pos.sl_price = pos.tp1_price
        pos.status = PositionStatus.TP2_HIT
        await self.state.save_position(pos)
        await self.state.push_log(
            "INFO",
            f"TP2 hit {pos.symbol} @ {price:.4f}, SL locked at TP1 {pos.tp1_price:.4f}, TP→TP3",
        )

        pos.sl_price = be_price
        pos.status = PositionStatus.TP1_HIT
        await self.state.save_position(pos)
        await self.state.push_log(
            "INFO",
            f"TP1 hit {pos.symbol} @ {price:.4f}, SL moved to breakeven {be_price:.4f}, TP→TP2",
        )

    async def _handle_tp2_hit(self, pos: Position, price: float):
        """TP2 hit: partially close, move SL above TP1 (lock in profit)."""
        log.info(f"TP2 hit {pos.symbol} @ {price:.4f}")
        close_qty = round(pos.qty * (self.cfg.tp2_close_pct / 100.0), 6)
        if close_qty > 0:
            await self.client.close_position(pos.symbol, pos.side, close_qty)

        # Move SL to TP1 level (guaranteed profit)
        await self.client.set_trading_stop(
            symbol=pos.symbol,
            side=pos.side,
            sl=pos.tp1_price,
            tp=pos.tp3_price,
        )

        pos.sl_price = pos.tp1_price
        pos.status = PositionStatus.TP2_HIT
        await self.state.save_position(pos)
        await self.state.push_log(
            "INFO",
            f"TP2 hit {pos.symbol} @ {price:.4f}, SL locked at TP1 {pos.tp1_price:.4f}, TP→TP3",
        )

    async def _handle_closed_externally(self, pos: Position):
        """Position no longer on exchange — SL or TP3 hit."""
        log.info(f"Position {pos.symbol} closed externally")
        pos.status = PositionStatus.CLOSED
        pos.closed_at = time.time()
        pos.close_reason = "external_close"
        await self.state.save_position(pos)
        journal.record_trade_close(pos)
        await self.state.push_log("INFO", f"Closed externally: {pos.symbol}")

    async def close_position(self, pos: Position, reason: str = "manual"):
        """Force-close a position at market."""
        try:
            live_pos = await self.client.get_positions()
            live_map = {p["symbol"]: p for p in live_pos}
            live = live_map.get(pos.symbol)
            qty = float(live.get("size", pos.qty)) if live else pos.qty

            if qty > 0:
                await self.client.close_position(pos.symbol, pos.side, qty)

            pos.status = PositionStatus.CLOSED
            pos.closed_at = time.time()
            pos.close_reason = reason
            await self.state.save_position(pos)
            journal.record_trade_close(pos)
            await self.state.push_log("INFO", f"Closed {pos.symbol} reason={reason}")
        except Exception as e:
            log.error(f"Close position error {pos.symbol}: {e}")

    async def close_all_positions(self, reason: str = "manual_halt"):
        """Emergency: close all open positions."""
        positions = await self.state.get_open_positions()
        for pos in positions:
            await self.close_position(pos, reason=reason)
        log.info(f"Closed all {len(positions)} positions")
