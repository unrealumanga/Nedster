"""
H2Wealth - Signal Engine
Non-traditional signals only:
  1. Order Flow Imbalance (OFI)      - orderbook pressure
  2. Cumulative Volume Delta (CVD)   - buy/sell aggressor delta
  3. Funding Rate Z-score           - funding anomaly vs 30-period mean
  4. Liquidation Heatmap Score      - proximity to major liq levels
  5. Regime Detection               - trend strength + microstructure
"""

from __future__ import annotations
import asyncio, hashlib, logging, math, time, uuid
from typing import Dict, List, Optional, Tuple
import numpy as np
from core.config import Config, Signal, Side
from core.bybit_client import BybitClient

log = logging.getLogger("signals")


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── 1. Order Flow Imbalance ──────────────────────────────────────────────────


def calc_ofi(orderbook: Dict, depth: int = 10) -> float:
    """
    OFI = (bid_volume_top_N - ask_volume_top_N) / total_volume_top_N
    Returns -1..+1; positive = buy pressure.
    """
    bids = orderbook.get("b", [])[:depth]
    asks = orderbook.get("a", [])[:depth]
    bid_vol = sum(_safe_float(b[1]) for b in bids)
    ask_vol = sum(_safe_float(a[1]) for a in asks)
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


# ── 2. Cumulative Volume Delta ────────────────────────────────────────────────


def calc_cvd(trades: List[Dict], lookback: int = 200) -> Tuple[float, float]:
    """
    CVD = sum of (buy_qty - sell_qty) over recent trades.
    Returns (cvd_value, cvd_momentum) - momentum = change in last 20% of period.
    Bybit trade side: 'Buy' means taker was buyer (market buy = aggressor).
    """
    recent = trades[:lookback]
    if not recent:
        return 0.0, 0.0

    deltas = []
    for t in recent:
        qty = _safe_float(t.get("size", 0))
        side = t.get("side", "")
        deltas.append(qty if side == "Buy" else -qty)

    cvd = sum(deltas)
    # Momentum: compare last 20% to first 80%
    cut = max(1, int(len(deltas) * 0.8))
    early = sum(deltas[:cut])
    late = sum(deltas[cut:])
    momentum = late - (early / cut) * (len(deltas) - cut) if cut > 0 else 0.0

    # Normalize by total volume
    total_vol = sum(abs(d) for d in deltas)
    norm_cvd = cvd / total_vol if total_vol > 0 else 0.0
    norm_mom = momentum / total_vol if total_vol > 0 else 0.0
    return norm_cvd, norm_mom


# ── 3. Funding Rate Z-score ───────────────────────────────────────────────────


def calc_funding_zscore(funding_history: List[Dict]) -> float:
    """
    Z-score of current funding rate vs recent 30 periods.
    Extreme positive funding = longs paying heavily = bearish pressure (short signal).
    Extreme negative = shorts paying = bullish pressure (long signal).
    Returns -3..+3 (clipped).
    """
    rates = []
    for f in funding_history:
        r = _safe_float(f.get("fundingRate", 0))
        rates.append(r)
    if len(rates) < 3:
        return 0.0
    arr = np.array(rates)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0.0
    z = (arr[0] - mean) / std  # arr[0] = most recent
    return float(np.clip(z, -3, 3))


# ── 4. Liquidation Heatmap ────────────────────────────────────────────────────


def calc_liq_score(ticker: Dict, orderbook: Dict) -> Tuple[float, Side]:
    """
    Approximate where stop-loss clusters might be by:
    - Recent high/low from 24h data
    - OI-weighted distance
    Returns (score 0..1, likely breakout direction).
    """
    try:
        price = _safe_float(ticker.get("lastPrice", 0))
        high24h = _safe_float(ticker.get("highPrice24h", 0))
        low24h = _safe_float(ticker.get("lowPrice24h", 0))
        oi = _safe_float(ticker.get("openInterest", 0))
        oi_value = _safe_float(ticker.get("openInterestValue", 0))

        if price <= 0 or high24h <= low24h:
            return 0.0, Side.BUY

        # Distance to key levels as % of range
        range_size = high24h - low24h
        dist_to_high = (high24h - price) / range_size
        dist_to_low = (price - low24h) / range_size

        # OI concentration proxy: high OI near extremes = liq hunt potential
        oi_pct = min(oi_value / 1e8, 1.0) if oi_value > 0 else 0.0  # normalize to $100M

        # If price near high, likely short stops above → upward sweep possible
        if dist_to_high < 0.15:
            score = (1.0 - dist_to_high) * (0.5 + 0.5 * oi_pct)
            return min(score, 1.0), Side.BUY
        elif dist_to_low < 0.15:
            score = (1.0 - dist_to_low) * (0.5 + 0.5 * oi_pct)
            return min(score, 1.0), Side.SELL
        else:
            return 0.1, Side.BUY
    except Exception:
        return 0.0, Side.BUY


# ── 5. Regime Detection ───────────────────────────────────────────────────────


def calc_regime(klines: List) -> Tuple[str, float, float]:
    """
    Detects market regime from OHLCV:
    - ADX-like directional strength
    - Volatility percentile
    - Microstructure (spread proxy)
    Returns (regime: 'trend_up'|'trend_down'|'range'|'volatile', strength 0..1, direction -1..1)
    """
    if len(klines) < 20:
        return "unknown", 0.0, 0.0

    # klines: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
    closes = np.array([_safe_float(k[4]) for k in klines[:50]])
    highs = np.array([_safe_float(k[2]) for k in klines[:50]])
    lows = np.array([_safe_float(k[3]) for k in klines[:50]])

    if len(closes) < 14:
        return "unknown", 0.0, 0.0

    # Simple directional strength (like ADX without wilder smoothing)
    tr = np.maximum(
        highs[:-1] - lows[:-1],
        np.maximum(np.abs(highs[:-1] - closes[1:]), np.abs(lows[:-1] - closes[1:])),
    )
    atr14 = np.mean(tr[:14]) if len(tr) >= 14 else np.mean(tr)

    dm_up = np.where(
        (highs[:-1] - highs[1:]) > (lows[1:] - lows[:-1]),
        np.maximum(highs[:-1] - highs[1:], 0),
        0,
    )
    dm_down = np.where(
        (lows[1:] - lows[:-1]) > (highs[:-1] - highs[1:]),
        np.maximum(lows[1:] - lows[:-1], 0),
        0,
    )

    di_up = np.mean(dm_up[:14]) / atr14 if atr14 > 0 else 0
    di_down = np.mean(dm_down[:14]) / atr14 if atr14 > 0 else 0
    dx = abs(di_up - di_down) / (di_up + di_down + 1e-9)
    adx = float(np.clip(dx, 0, 1))

    # Volatility
    returns = np.diff(np.log(closes + 1e-9))
    vol = float(np.std(returns) * math.sqrt(288))  # annualized for 5m
    vol_score = min(vol / 2.0, 1.0)  # 200% vol = max

    # Direction
    sma5 = float(np.mean(closes[:5]))
    sma20 = float(np.mean(closes[:20]))
    direction = 1.0 if sma5 > sma20 else -1.0

    if adx > 0.25 and vol_score < 0.7:
        regime = "trend_up" if direction > 0 else "trend_down"
    elif vol_score > 0.6:
        regime = "volatile"
    else:
        regime = "range"

    return regime, adx, direction


# ── Composite Scorer ──────────────────────────────────────────────────────────


class SignalEngine:
    WEIGHTS = {
        "ofi": 0.30,
        "cvd": 0.25,
        "funding": 0.20,
        "liq": 0.15,
        "regime": 0.10,
    }

    def __init__(self, cfg: Config, client: BybitClient):
        self.cfg = cfg
        self.client = client

    async def score_symbol(self, symbol: str, ticker: Dict) -> Optional[Signal]:
        """Score a single symbol. Returns Signal or None if not strong enough."""
        try:
            ob, trades, funding, klines = await asyncio.gather(
                self.client.get_orderbook(symbol, depth=20),
                self.client.get_recent_trades(symbol, limit=300),
                self.client.get_funding_rate(symbol),
                self.client.get_klines(symbol, interval="5", limit=50),
                return_exceptions=True,
            )

            # Bail on any fetch failure
            if any(isinstance(x, Exception) for x in [ob, trades, funding, klines]):
                return None

            price = _safe_float(ticker.get("lastPrice", 0))
            if price <= 0:
                return None

            # Compute individual signals
            ofi_raw = calc_ofi(ob)  # -1..+1
            cvd_raw, cvd_mom = calc_cvd(trades)  # -1..+1 each
            fund_z = calc_funding_zscore(funding)  # -3..+3
            liq_score, liq_side = calc_liq_score(ticker, ob)  # 0..1, side
            regime, adx, direction = calc_regime(klines)  # str, 0..1, -1..1

            # Normalize to 0..1 scores in direction of trade
            # OFI: positive = buy signal
            ofi_bull = (ofi_raw + 1) / 2  # 0..1
            ofi_bear = 1 - ofi_bull

            # CVD: positive = buy signal
            cvd_bull = (cvd_raw + 1) / 2 * (0.7 + 0.3 * ((cvd_mom + 1) / 2))
            cvd_bear = 1 - (cvd_raw + 1) / 2 * (0.7 + 0.3 * ((cvd_mom + 1) / 2))

            # Funding Z: large positive = shorts signal; large negative = longs
            fund_bear = min((fund_z + 3) / 6, 1.0)
            fund_bull = 1 - fund_bear

            # Regime: only trade in matching regime
            if regime in ("volatile",):
                return None  # skip volatile regimes
            regime_bull = (0.8 if direction > 0 else 0.2) * adx + 0.1
            regime_bear = 1 - regime_bull

            # Best direction
            bull_score = (
                self.WEIGHTS["ofi"] * ofi_bull
                + self.WEIGHTS["cvd"] * cvd_bull
                + self.WEIGHTS["funding"] * fund_bull
                + self.WEIGHTS["liq"] * (liq_score if liq_side == Side.BUY else 0)
                + self.WEIGHTS["regime"] * regime_bull
            )
            bear_score = (
                self.WEIGHTS["ofi"] * ofi_bear
                + self.WEIGHTS["cvd"] * cvd_bear
                + self.WEIGHTS["funding"] * fund_bear
                + self.WEIGHTS["liq"] * (liq_score if liq_side == Side.SELL else 0)
                + self.WEIGHTS["regime"] * regime_bear
            )

            if bull_score > bear_score:
                side, raw_score = Side.BUY, bull_score
            else:
                side, raw_score = Side.SELL, bear_score

            # Minimum threshold to be considered at all
            if raw_score < 0.55:
                return None

            score = raw_score * 100  # 0..100

            reasons = []
            if abs(ofi_raw) > 0.2:
                reasons.append(
                    f"OFI {'buy' if ofi_raw > 0 else 'sell'} pressure {ofi_raw:.2f}"
                )
            if abs(cvd_raw) > 0.1:
                reasons.append(f"CVD {'bull' if cvd_raw > 0 else 'bear'} {cvd_raw:.2f}")
            if abs(fund_z) > 1.5:
                reasons.append(f"Funding Z={fund_z:.1f}")
            if liq_score > 0.5:
                reasons.append(f"Liq hunt {liq_side.value} zone")
            if adx > 0.3:
                reasons.append(f"Regime {regime} ADX={adx:.2f}")

            now = time.time()
            sig = Signal(
                symbol=symbol,
                side=side,
                score=round(score, 2),
                entry_price=price,
                created_at=now,
                expires_at=now + self.cfg.signal_expiry_sec,
                reason=" | ".join(reasons) or "Composite signal",
                ofi_score=round(ofi_bull if side == Side.BUY else ofi_bear, 3),
                cvd_score=round(cvd_bull if side == Side.BUY else cvd_bear, 3),
                funding_score=round(fund_bull if side == Side.BUY else fund_bear, 3),
                liq_score=round(liq_score, 3),
                regime_score=round(regime_bull if side == Side.BUY else regime_bear, 3),
                signal_id=f"sig_{symbol}_{int(now)}_{uuid.uuid4().hex[:6]}",
            )
            return sig

        except Exception as e:
            log.debug(f"Score error {symbol}: {e}")
            return None

    async def scan_all(self, symbols: List[str]) -> List[Signal]:
        """Scan all symbols concurrently (batched to respect rate limits)."""
        tickers_raw = await self.client.get_tickers()
        ticker_map = {t["symbol"]: t for t in tickers_raw}

        # Filter to USDT perpetuals with decent volume
        eligible = []
        for sym in symbols:
            t = ticker_map.get(sym, {})
            vol24h = _safe_float(t.get("turnover24h", 0))
            if vol24h > 5_000_000:  # $5M+ daily volume only
                eligible.append((sym, t))

        log.info(f"Scanning {len(eligible)} eligible pairs...")

        BATCH = 10
        signals = []
        for i in range(0, len(eligible), BATCH):
            batch = eligible[i : i + BATCH]
            results = await asyncio.gather(
                *[self.score_symbol(sym, ticker) for sym, ticker in batch],
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, Signal):
                    signals.append(r)
            await asyncio.sleep(0.5)  # brief pause between batches

        # Sort by score, return top N
        signals.sort(key=lambda s: s.score, reverse=True)
        top = signals[: self.cfg.top_signals_n]
        log.info(f"Found {len(signals)} signals, top {len(top)} selected")
        return top
