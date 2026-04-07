"""
H2Wealth - Core Configuration & Shared Types
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Side(str, Enum):
    BUY = "Buy"
    SELL = "Sell"

class PositionStatus(str, Enum):
    PENDING   = "pending"
    OPEN      = "open"
    TP1_HIT   = "tp1_hit"
    TP2_HIT   = "tp2_hit"
    CLOSED    = "closed"
    EXPIRED   = "expired"
    SL_HIT    = "sl_hit"
    ERROR     = "error"

class BotStatus(str, Enum):
    RUNNING   = "running"
    PAUSED    = "paused"
    STOPPED   = "stopped"

@dataclass
class Config:
    # Bybit
    api_key:              str   = field(default_factory=lambda: os.getenv("BYBIT_API_KEY",""))
    api_secret:           str   = field(default_factory=lambda: os.getenv("BYBIT_API_SECRET",""))
    demo:                 bool  = field(default_factory=lambda: os.getenv("BYBIT_DEMO","true").lower()=="true")
    base_url:             str   = field(default_factory=lambda: os.getenv("BYBIT_BASE_URL","https://api-demo.bybit.com"))

    # Risk
    max_position_size_pct: float = field(default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE_PCT","5.0")))
    max_concurrent:        int   = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_POSITIONS","5")))
    sl_pct:                float = field(default_factory=lambda: float(os.getenv("SL_PCT","0.8")))
    tp1_pct:               float = field(default_factory=lambda: float(os.getenv("TP1_PCT","1.2")))
    tp2_pct:               float = field(default_factory=lambda: float(os.getenv("TP2_PCT","2.0")))
    tp3_pct:               float = field(default_factory=lambda: float(os.getenv("TP3_PCT","3.5")))
    tp1_close_pct:         float = field(default_factory=lambda: float(os.getenv("TP1_CLOSE_PCT","40")))
    tp2_close_pct:         float = field(default_factory=lambda: float(os.getenv("TP2_CLOSE_PCT","40")))
    tp3_close_pct:         float = field(default_factory=lambda: float(os.getenv("TP3_CLOSE_PCT","20")))
    leverage:              int   = field(default_factory=lambda: int(os.getenv("LEVERAGE","5")))
    min_trade_usdt:        float = field(default_factory=lambda: float(os.getenv("MIN_TRADE_USDT","10.0")))

    # Signals
    scan_interval_sec:     int   = field(default_factory=lambda: int(os.getenv("SCAN_INTERVAL_SEC","60")))
    top_signals_n:         int   = field(default_factory=lambda: int(os.getenv("TOP_SIGNALS_N","10")))
    signal_expiry_sec:     int   = field(default_factory=lambda: int(os.getenv("SIGNAL_EXPIRY_SEC","900")))
    signal_expire_profit_pct: float = field(default_factory=lambda: float(os.getenv("SIGNAL_EXPIRE_PROFIT_TAKE_PCT","75")))

    # Redis
    redis_host:            str   = field(default_factory=lambda: os.getenv("REDIS_HOST","localhost"))
    redis_port:            int   = field(default_factory=lambda: int(os.getenv("REDIS_PORT","6379")))
    redis_db:              int   = field(default_factory=lambda: int(os.getenv("REDIS_DB","0")))

    # WebUI
    webui_host:            str   = field(default_factory=lambda: os.getenv("WEBUI_HOST","0.0.0.0"))
    webui_port:            int   = field(default_factory=lambda: int(os.getenv("WEBUI_PORT","8080")))
    webui_secret:          str   = field(default_factory=lambda: os.getenv("WEBUI_SECRET","changeme"))

    # Logging
    log_level:             str   = field(default_factory=lambda: os.getenv("LOG_LEVEL","INFO"))
    log_file:              str   = field(default_factory=lambda: os.getenv("LOG_FILE","logs/h2wealth.log"))


@dataclass
class Signal:
    symbol:       str
    side:         Side
    score:        float          # composite 0-100
    entry_price:  float
    created_at:   float          # unix timestamp
    expires_at:   float          # unix timestamp
    reason:       str            # human-readable signal reason
    ofi_score:    float = 0.0
    cvd_score:    float = 0.0
    funding_score: float = 0.0
    liq_score:    float = 0.0
    regime_score: float = 0.0
    signal_id:    str = ""

    def ttl_pct(self, now: float) -> float:
        total = self.expires_at - self.created_at
        if total <= 0:
            return 100.0
        return ((now - self.created_at) / total) * 100.0

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at


@dataclass
class Position:
    position_id:   str
    symbol:        str
    side:          Side
    entry_price:   float
    qty:           float
    leverage:      int
    sl_price:      float
    tp1_price:     float
    tp2_price:     float
    tp3_price:     float
    signal_id:     str
    opened_at:     float
    signal_expires_at: float
    status:        PositionStatus = PositionStatus.OPEN
    bybit_order_id: str = ""
    sl_order_id:   str = ""
    tp1_order_id:  str = ""
    tp2_order_id:  str = ""
    tp3_order_id:  str = ""
    pnl_usdt:      float = 0.0
    closed_at:     Optional[float] = None
    close_reason:  str = ""

    def ttl_pct(self, now: float) -> float:
        total = self.signal_expires_at - self.opened_at
        if total <= 0:
            return 100.0
        return ((now - self.opened_at) / total) * 100.0
