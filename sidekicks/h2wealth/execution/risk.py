"""
H2Wealth - Risk Engine
Position sizing, TP/SL price calculation, equity guards.
"""
from __future__ import annotations
import logging, math
from dataclasses import dataclass
from typing import Optional, Tuple
from core.config import Config, Signal, Side

log = logging.getLogger("risk")


@dataclass
class TradeParams:
    symbol:     str
    side:       Side
    entry:      float
    qty:        float
    notional:   float
    sl_price:   float
    tp1_price:  float
    tp2_price:  float
    tp3_price:  float
    leverage:   int
    margin_req: float


class RiskEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute_trade(
        self,
        signal:     Signal,
        equity:     float,
        open_count: int,
        qty_step:   float = 0.001,
        price_step: float = 0.01,
        min_qty:    float = 0.001,
    ) -> Optional[TradeParams]:
        """
        Given a signal and current account state, compute full trade parameters.
        Returns None if trade should not be taken (risk guard failed).
        """
        # Guard: too many positions
        if open_count >= self.cfg.max_concurrent:
            log.info(f"Skip {signal.symbol}: at max concurrent {self.cfg.max_concurrent}")
            return None

        # Guard: equity too low
        if equity < self.cfg.min_trade_usdt * 2:
            log.warning(f"Skip {signal.symbol}: equity {equity:.2f} too low")
            return None

        # Position size as % of equity, capped
        risk_per_trade  = equity * (self.cfg.max_position_size_pct / 100.0)
        # Scale down if more positions open (equal weighting)
        slots_remaining = self.cfg.max_concurrent - open_count
        allocation      = equity / max(slots_remaining, 1) * (self.cfg.max_position_size_pct / 100.0)
        notional        = min(risk_per_trade, allocation)
        notional        = max(notional, self.cfg.min_trade_usdt)

        entry    = signal.entry_price
        leverage = self.cfg.leverage

        # Qty
        raw_qty  = (notional * leverage) / entry
        qty      = math.floor(raw_qty / qty_step) * qty_step
        qty      = max(qty, min_qty)

        # Actual notional after rounding
        actual_notional = qty * entry / leverage
        margin_req      = actual_notional

        # Guard: margin would exceed per-trade allocation
        if actual_notional > notional * 1.1:
            qty = math.floor(notional * leverage / entry / qty_step) * qty_step

        if qty < min_qty:
            log.info(f"Skip {signal.symbol}: qty {qty} below min {min_qty}")
            return None

        # TP / SL prices
        sl_pct  = self.cfg.sl_pct  / 100.0
        tp1_pct = self.cfg.tp1_pct / 100.0
        tp2_pct = self.cfg.tp2_pct / 100.0
        tp3_pct = self.cfg.tp3_pct / 100.0

        if signal.side == Side.BUY:
            sl    = self._round(entry * (1 - sl_pct),  price_step)
            tp1   = self._round(entry * (1 + tp1_pct), price_step)
            tp2   = self._round(entry * (1 + tp2_pct), price_step)
            tp3   = self._round(entry * (1 + tp3_pct), price_step)
        else:
            sl    = self._round(entry * (1 + sl_pct),  price_step)
            tp1   = self._round(entry * (1 - tp1_pct), price_step)
            tp2   = self._round(entry * (1 - tp2_pct), price_step)
            tp3   = self._round(entry * (1 - tp3_pct), price_step)

        return TradeParams(
            symbol    = signal.symbol,
            side      = signal.side,
            entry     = entry,
            qty       = round(qty, 6),
            notional  = round(actual_notional, 2),
            sl_price  = sl,
            tp1_price = tp1,
            tp2_price = tp2,
            tp3_price = tp3,
            leverage  = leverage,
            margin_req= round(margin_req, 2),
        )

    def breakeven_price(self, entry: float, side: Side, fee_pct: float = 0.055) -> float:
        """Price where trade is at breakeven including round-trip fees."""
        fee = fee_pct / 100.0
        if side == Side.BUY:
            return entry * (1 + fee * 2)
        return entry * (1 - fee * 2)

    @staticmethod
    def _round(price: float, step: float) -> float:
        return round(round(price / step) * step, 8)

    def get_instrument_params(self, instruments: list, symbol: str) -> Tuple[float, float, float]:
        """Extract qty_step, price_step, min_qty from instrument info."""
        for inst in instruments:
            if inst.get("symbol") == symbol:
                lot = inst.get("lotSizeFilter", {})
                price_filter = inst.get("priceFilter", {})
                qty_step    = _safe_float(lot.get("qtyStep", "0.001"))
                min_qty     = _safe_float(lot.get("minOrderQty", "0.001"))
                price_step  = _safe_float(price_filter.get("tickSize", "0.01"))
                return qty_step, price_step, min_qty
        return 0.001, 0.01, 0.001


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
