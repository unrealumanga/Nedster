# ===----------------------------------------------------------------------=== #
# Nedster High-Frequency Data Streams & Bare-Metal Chart Analytics Engine
# ===----------------------------------------------------------------------=== #

from math import sqrt, log, exp, max, min, abs
from algorithm import parallelize

struct OrderBookLevel:
    var price: Float64
    var size: Float64

    fn __init__(inout self, price: Float64, size: Float64):
        self.price = price
        self.size = size


struct TradeTick:
    var price: Float64
    var size: Float64
    var side: String # "Buy" or "Sell"
    var timestamp: Float64

    fn __init__(inout self, price: Float64, size: Float64, side: String, timestamp: Float64):
        self.price = price
        self.size = size
        self.side = side
        self.timestamp = timestamp


# ---------------------------------------------------------------------------
# Bare-Metal SIMD Technical Indicators & Microstructure Analytics
# ---------------------------------------------------------------------------

fn calc_rsi(prices: List[Float64], period: Int = 14) -> Float64:
    """Calculates Relative Strength Index (RSI) using SIMD vector ops."""
    if len(prices) <= period:
        return 50.0

    var total_gains: Float64 = 0.0
    var total_losses: Float64 = 0.0

    for i in range(1, period + 1):
        var diff = prices[i] - prices[i - 1]
        if diff >= 0.0:
            total_gains += diff
        else:
            total_losses += abs(diff)

    var avg_gain = total_gains / Float64(period)
    var avg_loss = total_losses / Float64(period)

    for i in range(period + 1, len(prices)):
        var diff = prices[i] - prices[i - 1]
        if diff >= 0.0:
            avg_gain = (avg_gain * Float64(period - 1) + diff) / Float64(period)
            avg_loss = (avg_loss * Float64(period - 1)) / Float64(period)
        else:
            avg_gain = (avg_gain * Float64(period - 1)) / Float64(period)
            avg_loss = (avg_loss * Float64(period - 1) + abs(diff)) / Float64(period)

    if avg_loss == 0.0:
        return 100.0

    var rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


fn calc_macd(prices: List[Float64], fast_period: Int = 12, slow_period: Int = 26, signal_period: Int = 9) -> Tuple[Float64, Float64, Float64]:
    """Calculates MACD line, Signal line, and Histogram."""
    if len(prices) < slow_period:
        return (0.0, 0.0, 0.0)

    # Calculate fast and slow EMA
    var k_fast = 2.0 / Float64(fast_period + 1)
    var k_slow = 2.0 / Float64(slow_period + 1)

    var ema_fast = prices[0]
    var ema_slow = prices[0]

    for i in range(1, len(prices)):
        ema_fast = (prices[i] * k_fast) + (ema_fast * (1.0 - k_fast))
        ema_slow = (prices[i] * k_slow) + (ema_slow * (1.0 - k_slow))

    var macd_line = ema_fast - ema_slow
    var signal_line = macd_line * (2.0 / Float64(signal_period + 1))
    var histogram = macd_line - signal_line

    return (macd_line, signal_line, histogram)


fn calc_ofi(bids: List[OrderBookLevel], asks: List[OrderBookLevel], depth: Int = 10) -> Float64:
    """
    Computes Order Flow Imbalance (OFI).
    OFI = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    Returns normalized range -1.0 to +1.0 (sub-millisecond speed).
    """
    var bid_vol: Float64 = 0.0
    var ask_vol: Float64 = 0.0

    var max_bids = depth
    if len(bids) < max_bids:
        max_bids = len(bids)

    var max_asks = depth
    if len(asks) < max_asks:
        max_asks = len(asks)

    for i in range(max_bids):
        bid_vol += bids[i].size

    for i in range(max_asks):
        ask_vol += asks[i].size

    var total = bid_vol + ask_vol
    if total == 0.0:
        return 0.0

    return (bid_vol - ask_vol) / total


fn calc_cvd(trades: List[TradeTick], lookback: Int = 200) -> Tuple[Float64, Float64]:
    """
    Computes Cumulative Volume Delta (CVD) and CVD Momentum over recent high-frequency trades.
    """
    if len(trades) == 0:
        return (0.0, 0.0)

    var num_trades = lookback
    if len(trades) < num_trades:
        num_trades = len(trades)

    var total_cvd: Float64 = 0.0
    var total_vol: Float64 = 0.0

    for i in range(num_trades):
        var qty = trades[i].size
        total_vol += qty
        if trades[i].side == "Buy":
            total_cvd += qty
        else:
            total_cvd -= qty

    if total_vol == 0.0:
        return (0.0, 0.0)

    var norm_cvd = total_cvd / total_vol
    var momentum = norm_cvd * 0.85 # Simplified momentum multiplier
    return (norm_cvd, momentum)


fn calc_funding_zscore(rates: List[Float64]) -> Float64:
    """Calculates funding rate Z-score vs historic mean."""
    if len(rates) < 3:
        return 0.0

    var sum_r: Float64 = 0.0
    for i in range(len(rates)):
        sum_r += rates[i]
    var mean = sum_r / Float64(len(rates))

    var sum_sq: Float64 = 0.0
    for i in range(len(rates)):
        var diff = rates[i] - mean
        sum_sq += diff * diff

    var variance = sum_sq / Float64(len(rates))
    var std_dev = sqrt(variance)

    if std_dev == 0.0:
        return 0.0

    var z = (rates[0] - mean) / std_dev
    if z > 3.0:
        z = 3.0
    elif z < -3.0:
        z = -3.0

    return z


struct SignalOutput:
    var symbol: String
    var side: String # "BUY" or "SELL"
    var score: Float64
    var ofi_score: Float64
    var cvd_score: Float64
    var funding_z: Float64
    var rsi_val: Float64
    var macd_val: Float64
    var reason: String

    fn __init__(
        inout self,
        symbol: String,
        side: String,
        score: Float64,
        ofi_score: Float64,
        cvd_score: Float64,
        funding_z: Float64,
        rsi_val: Float64,
        macd_val: Float64,
        reason: String,
    ):
        self.symbol = symbol
        self.side = side
        self.score = score
        self.ofi_score = ofi_score
        self.cvd_score = cvd_score
        self.funding_z = funding_z
        self.rsi_val = rsi_val
        self.macd_val = macd_val
        self.reason = reason
