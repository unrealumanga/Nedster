"""
H2Wealth - Bybit v5 API Client
Handles REST + WebSocket with built-in rate limiting (120 req/min).
"""

from __future__ import annotations
import asyncio, hashlib, hmac, json, logging, time
from collections import deque
from typing import Any, Dict, List, Optional
import aiohttp
from core.config import Config, Side

log = logging.getLogger("bybit")


class RateLimiter:
    """Token-bucket: max 120 req/min = 2/sec with burst headroom."""

    def __init__(self, max_per_min: int = 100):  # stay 20 under hard limit
        self._interval = 60.0 / max_per_min
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


class BybitClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._rl = RateLimiter(100)
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        self._session = aiohttp.ClientSession(
            base_url=self.cfg.base_url, timeout=aiohttp.ClientTimeout(total=10)
        )

    async def stop(self):
        if self._session:
            await self._session.close()

    def _sign(self, params: str, ts: int) -> str:
        recv = 5000
        to_sign = f"{ts}{self.cfg.api_key}{recv}{params}"
        return hmac.new(
            self.cfg.api_secret.encode(), to_sign.encode(), hashlib.sha256
        ).hexdigest()

    async def _get(self, path: str, params: Dict = None, auth: bool = False) -> Dict:
        await self._rl.acquire()
        params = params or {}
        headers = {"Content-Type": "application/json"}
        if auth:
            ts = int(time.time() * 1000)
            qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            headers.update(
                {
                    "X-BAPI-API-KEY": self.cfg.api_key,
                    "X-BAPI-TIMESTAMP": str(ts),
                    "X-BAPI-RECV-WINDOW": "5000",
                    "X-BAPI-SIGN": self._sign(qs, ts),
                }
            )
        async with self._session.get(path, params=params, headers=headers) as r:
            data = await r.json()
            if data.get("retCode", 0) != 0:
                log.error(f"Bybit GET {path} error: {data.get('retMsg')} | {params}")
            return data

    async def _post(self, path: str, body: Dict, auth: bool = True) -> Dict:
        await self._rl.acquire()
        ts = int(time.time() * 1000)
        body_str = json.dumps(body)
        headers = {
            "Content-Type": "application/json",
            "X-BAPI-API-KEY": self.cfg.api_key,
            "X-BAPI-TIMESTAMP": str(ts),
            "X-BAPI-RECV-WINDOW": "5000",
            "X-BAPI-SIGN": self._sign(body_str, ts),
        }
        async with self._session.post(path, data=body_str, headers=headers) as r:
            data = await r.json()
            if data.get("retCode", 0) != 0:
                log.error(f"Bybit POST {path} error: {data.get('retMsg')} | {body}")
            return data

    # ── Market Data ──────────────────────────────────────────────────────────

    async def get_instruments(self) -> List[Dict]:
        """All linear perpetual instruments."""
        r = await self._get(
            "/v5/market/instruments-info", {"category": "linear", "limit": "1000"}
        )
        return r.get("result", {}).get("list", [])

    async def get_tickers(self) -> List[Dict]:
        """All linear tickers (price, volume, funding, OI)."""
        r = await self._get("/v5/market/tickers", {"category": "linear"})
        return r.get("result", {}).get("list", [])

    async def get_klines(
        self, symbol: str, interval: str = "5", limit: int = 200
    ) -> List:
        r = await self._get(
            "/v5/market/kline",
            {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": str(limit),
            },
        )
        return r.get("result", {}).get("list", [])

    async def get_orderbook(self, symbol: str, depth: int = 50) -> Dict:
        r = await self._get(
            "/v5/market/orderbook",
            {"category": "linear", "symbol": symbol, "limit": str(depth)},
        )
        return r.get("result", {})

    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List:
        r = await self._get(
            "/v5/market/recent-trade",
            {"category": "linear", "symbol": symbol, "limit": str(limit)},
        )
        return r.get("result", {}).get("list", [])

    async def get_funding_rate(self, symbol: str) -> Dict:
        r = await self._get(
            "/v5/market/funding/history",
            {"category": "linear", "symbol": symbol, "limit": "10"},
        )
        return r.get("result", {}).get("list", [])

    async def get_open_interest(
        self, symbol: str, interval: str = "5min", limit: int = 50
    ) -> List:
        r = await self._get(
            "/v5/market/open-interest",
            {
                "category": "linear",
                "symbol": symbol,
                "intervalTime": interval,
                "limit": str(limit),
            },
        )
        return r.get("result", {}).get("list", [])

    # ── Account ──────────────────────────────────────────────────────────────

    async def get_wallet_balance(self, coin: str = "USDT") -> float:
        r = await self._get(
            "/v5/account/wallet-balance", {"accountType": "UNIFIED"}, auth=True
        )
        for acc in r.get("result", {}).get("list", []):
            for c in acc.get("coin", []):
                if c.get("coin") == coin:
                    val = (
                        c.get("equity")
                        or c.get("walletBalance")
                        or c.get("availableToWithdraw", "0")
                    )
                    return float(val) if val else 0.0
        return 0.0

    async def get_positions(self) -> List[Dict]:
        r = await self._get(
            "/v5/position/list", {"category": "linear", "settleCoin": "USDT"}, auth=True
        )
        return r.get("result", {}).get("list", [])

    # ── Orders ───────────────────────────────────────────────────────────────

    async def place_order(
        self,
        symbol: str,
        side: Side,
        qty: float,
        order_type: str = "Market",
        price: float = None,
        reduce_only: bool = False,
        close_on_trigger: bool = False,
        sl: float = None,
        tp: float = None,
        order_link_id: str = "",
    ) -> Dict:
        body: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "side": side.value,
            "orderType": order_type,
            "qty": str(round(qty, 6)),
            "reduceOnly": reduce_only,
            "closeOnTrigger": close_on_trigger,
            "timeInForce": "GTC" if order_type == "Limit" else "IOC",
        }
        if price:
            body["price"] = str(round(price, 6))
        if sl:
            body["stopLoss"] = str(round(sl, 6))
            body["slTriggerBy"] = "LastPrice"
        if tp:
            body["takeProfit"] = str(round(tp, 6))
            body["tpTriggerBy"] = "LastPrice"
        if order_link_id:
            body["orderLinkId"] = order_link_id
        return await self._post("/v5/order/create", body)

    async def amend_order(
        self, symbol: str, order_id: str, sl: float = None, tp: float = None
    ) -> Dict:
        body: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }
        if sl:
            body["stopLoss"] = str(round(sl, 6))
        if tp:
            body["takeProfit"] = str(round(tp, 6))
        return await self._post("/v5/order/amend", body)

    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        return await self._post(
            "/v5/order/cancel",
            {"category": "linear", "symbol": symbol, "orderId": order_id},
        )

    async def cancel_all_orders(self, symbol: str) -> Dict:
        return await self._post(
            "/v5/order/cancel-all", {"category": "linear", "symbol": symbol}
        )

    async def set_leverage(self, symbol: str, leverage: int) -> Dict:
        return await self._post(
            "/v5/position/set-leverage",
            {
                "category": "linear",
                "symbol": symbol,
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage),
            },
        )

    async def set_trading_stop(
        self,
        symbol: str,
        side: Side,
        sl: float = None,
        tp: float = None,
        tsl_pct: float = None,
    ) -> Dict:
        body: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": 0,
        }
        if sl:
            body["stopLoss"] = str(round(sl, 6))
            body["slTriggerBy"] = "LastPrice"
        if tp:
            body["takeProfit"] = str(round(tp, 6))
            body["tpTriggerBy"] = "LastPrice"
        return await self._post("/v5/position/trading-stop", body)

    async def close_position(self, symbol: str, side: Side, qty: float) -> Dict:
        close_side = Side.SELL if side == Side.BUY else Side.BUY
        return await self.place_order(
            symbol, close_side, qty, order_type="Market", reduce_only=True
        )

    async def ping(self) -> bool:
        try:
            r = await self._get("/v5/market/time")
            return r.get("retCode", -1) == 0
        except Exception as e:
            log.error(f"Ping failed: {e}")
            return False
