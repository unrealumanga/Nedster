"""
H2Wealth - SQLite Trade Journal
Permanent record of all trades, signals, and performance.
"""
from __future__ import annotations
import asyncio, json, logging, sqlite3, time
from pathlib import Path
from typing import Dict, List, Optional
from core.config import Position, Signal

log = logging.getLogger("journal")
DB_PATH = Path("data/journal.db")


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            position_id     TEXT PRIMARY KEY,
            symbol          TEXT NOT NULL,
            side            TEXT NOT NULL,
            entry_price     REAL,
            qty             REAL,
            leverage        INTEGER,
            sl_price        REAL,
            tp1_price       REAL,
            tp2_price       REAL,
            tp3_price       REAL,
            signal_id       TEXT,
            opened_at       REAL,
            closed_at       REAL,
            pnl_usdt        REAL,
            status          TEXT,
            close_reason    TEXT,
            signal_score    REAL,
            signal_reason   TEXT,
            created_at      REAL DEFAULT (unixepoch())
        );

        CREATE TABLE IF NOT EXISTS signals (
            signal_id       TEXT PRIMARY KEY,
            symbol          TEXT,
            side            TEXT,
            score           REAL,
            ofi_score       REAL,
            cvd_score       REAL,
            funding_score   REAL,
            liq_score       REAL,
            regime_score    REAL,
            entry_price     REAL,
            reason          TEXT,
            created_at      REAL,
            expires_at      REAL,
            acted_on        INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS performance_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              REAL,
            equity          REAL,
            open_positions  INTEGER,
            total_trades    INTEGER,
            win_trades      INTEGER,
            total_pnl       REAL
        );

        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_opened ON trades(opened_at);
        CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
        """)
    log.info("Journal DB initialized")


def record_signal(sig: Signal, acted_on: bool = False):
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO signals
            (signal_id,symbol,side,score,ofi_score,cvd_score,funding_score,liq_score,
             regime_score,entry_price,reason,created_at,expires_at,acted_on)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (sig.signal_id, sig.symbol, sig.side.value if hasattr(sig.side,'value') else sig.side,
              sig.score, sig.ofi_score, sig.cvd_score, sig.funding_score,
              sig.liq_score, sig.regime_score, sig.entry_price, sig.reason,
              sig.created_at, sig.expires_at, int(acted_on)))


def record_trade_open(pos: Position, signal: Optional[Signal] = None):
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO trades
            (position_id,symbol,side,entry_price,qty,leverage,sl_price,
             tp1_price,tp2_price,tp3_price,signal_id,opened_at,status,
             signal_score,signal_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (pos.position_id, pos.symbol,
              pos.side.value if hasattr(pos.side,'value') else pos.side,
              pos.entry_price, pos.qty, pos.leverage,
              pos.sl_price, pos.tp1_price, pos.tp2_price, pos.tp3_price,
              pos.signal_id, pos.opened_at, pos.status.value if hasattr(pos.status,'value') else pos.status,
              signal.score if signal else None,
              signal.reason if signal else None))


def record_trade_close(pos: Position):
    with get_conn() as conn:
        conn.execute("""
            UPDATE trades SET closed_at=?,pnl_usdt=?,status=?,close_reason=?
            WHERE position_id=?
        """, (pos.closed_at or time.time(),
              pos.pnl_usdt, pos.status.value if hasattr(pos.status,'value') else pos.status,
              pos.close_reason, pos.position_id))


def snapshot_performance(equity: float, open_pos: int):
    with get_conn() as conn:
        row = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN pnl_usdt>0 THEN 1 ELSE 0 END) as wins,
                   SUM(pnl_usdt) as total_pnl
            FROM trades WHERE closed_at IS NOT NULL
        """).fetchone()
        conn.execute("""
            INSERT INTO performance_snapshots(ts,equity,open_positions,total_trades,win_trades,total_pnl)
            VALUES (?,?,?,?,?,?)
        """, (time.time(), equity, open_pos, row["total"] or 0, row["wins"] or 0, row["total_pnl"] or 0.0))


def get_performance_summary() -> Dict:
    with get_conn() as conn:
        row = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN pnl_usdt>0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN pnl_usdt<=0 THEN 1 ELSE 0 END) as losses,
                   SUM(pnl_usdt) as total_pnl,
                   AVG(pnl_usdt) as avg_pnl,
                   MAX(pnl_usdt) as best_trade,
                   MIN(pnl_usdt) as worst_trade
            FROM trades WHERE closed_at IS NOT NULL
        """).fetchone()
        return dict(row) if row else {}


def get_recent_trades(limit: int = 50) -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM trades ORDER BY opened_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
