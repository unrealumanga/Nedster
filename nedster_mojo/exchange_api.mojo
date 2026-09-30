# ===----------------------------------------------------------------------=== #
# Nedster Zero-Overhead Python Interop Exchange API & WebSocket Listener
# ===----------------------------------------------------------------------=== #

from python import Python
from .chart_analytics import OrderBookLevel, TradeTick, SignalOutput, calc_ofi, calc_cvd, calc_rsi, calc_macd

struct ExchangeClient:
    var exchange_id: String
    var api_key: String
    var api_secret: String

    fn __init__(inout self, exchange_id: String = "binance", api_key: String = "", api_secret: String = ""):
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret

    fn fetch_ticker_price(self, symbol: String) raises -> Float64:
        """Imports Python ccxt library to fetch instant ticker quote."""
        let ccxt = Python.import_module("ccxt")
        var ex_class = getattr(ccxt, self.exchange_id)
        
        var config = Python.dict()
        if len(self.api_key) > 0:
            config["apiKey"] = self.api_key
            config["secret"] = self.api_secret

        var ex_instance = ex_class(config)
        var ticker = ex_instance.fetch_ticker(symbol)
        return Float64(ticker["last"])

    fn fetch_orderbook_and_analyze(self, symbol: String) raises -> SignalOutput:
        """Fetches live exchange orderbook, converts to Mojo structs, and computes signals."""
        let ccxt = Python.import_module("ccxt")
        var ex_class = getattr(ccxt, self.exchange_id)
        var ex_instance = ex_class()

        var ob_py = ex_instance.fetch_order_book(symbol)
        var bids_py = ob_py["bids"]
        var asks_py = ob_py["asks"]

        var bids = List[OrderBookLevel]()
        var asks = List[OrderBookLevel]()

        var num_bids = Int(len(bids_py))
        if num_bids > 20:
            num_bids = 20
        for i in range(num_bids):
            var level = bids_py[i]
            bids.append(OrderBookLevel(Float64(level[0]), Float64(level[1])))

        var num_asks = Int(len(asks_py))
        if num_asks > 20:
            num_asks = 20
        for i in range(num_asks):
            var level = asks_py[i]
            asks.append(OrderBookLevel(Float64(level[0]), Float64(level[1])))

        var ofi_score = calc_ofi(bids, asks, 10)

        # Fetch recent trades
        var trades_py = ex_instance.fetch_trades(symbol, limit=100)
        var trades = List[TradeTick]()
        for i in range(Int(len(trades_py))):
            var t = trades_py[i]
            var side_str = String(t.get("side", "buy")).capitalize()
            trades.append(TradeTick(Float64(t.get("price", 0.0)), Float64(t.get("amount", 0.0)), side_str, Float64(t.get("timestamp", 0.0))))

        var cvd_res = calc_cvd(trades, 100)
        var cvd_score = cvd_res.0

        # Fetch klines for RSI and MACD
        var ohlcv_py = ex_instance.fetch_ohlcv(symbol, timeframe="1m", limit=30)
        var close_prices = List[Float64]()
        for i in range(Int(len(ohlcv_py))):
            close_prices.append(Float64(ohlcv_py[i][4]))

        var rsi_val = calc_rsi(close_prices, 14)
        var macd_tuple = calc_macd(close_prices, 12, 26, 9)
        var macd_val = macd_tuple.0

        var side = "BUY"
        var overall_score = (ofi_score * 0.4 + cvd_score * 0.4) * 100.0
        if overall_score < 0.0:
            side = "SELL"
            overall_score = abs(overall_score)

        return SignalOutput(
            symbol=symbol,
            side=side,
            score=overall_score,
            ofi_score=ofi_score,
            cvd_score=cvd_score,
            funding_z=0.0,
            rsi_val=rsi_val,
            macd_val=macd_val,
            reason="CCXT Real-Time Stream Imbalance Analysis",
        )
