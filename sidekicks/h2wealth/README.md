# ⟁ H2Wealth — Automated Crypto Trading Bot

**Bybit v5 · Non-traditional signals · Full TP cascade · Auto expiry guardian**

---

## Architecture

```
Market Data (Go-class async) → Signal Engine → Pair Ranker → Risk Engine → Execution → Monitor
                                     ↓
                              Redis State Store
                                     ↓
                          WebUI (FastAPI) | TUI (Textual)
```

## Signals (Non-Traditional)

| Signal | Description | Weight |
|--------|-------------|--------|
| **OFI** | Order Flow Imbalance — bid vs ask volume pressure at top of book | 30% |
| **CVD** | Cumulative Volume Delta — buy vs sell aggressor momentum | 25% |
| **Funding Z-score** | Funding rate anomaly vs 30-period mean | 20% |
| **Liq Heatmap** | Proximity to stop-cluster zones (24h high/low + OI) | 15% |
| **Regime** | ADX-based trend strength + microstructure direction | 10% |

All 155 pairs are scanned every `SCAN_INTERVAL_SEC` seconds. Only top `TOP_SIGNALS_N` are acted upon.

## Position Lifecycle

```
Open (market order + SL)
  ↓
TP1 hit → close 40% → move SL to breakeven+fees → TP target → TP2
  ↓
TP2 hit → close 40% → move SL to TP1 (locked profit) → TP target → TP3
  ↓
TP3 hit → position fully closed by exchange
  OR
Signal TTL 75% elapsed + PNL > 0 → force close (expiry guardian)
Signal TTL 100% elapsed → force close regardless
```

## Quick Start

### 1. Get Bybit Demo API Keys
1. Go to https://testnet.bybit.com (demo) or https://bybit.com (live)
2. Account → API Management → Create API Key
3. Permissions: **Contract Trading** (read + trade)

### 2. Configure
```bash
cp .env.example .env
nano .env   # set BYBIT_API_KEY and BYBIT_API_SECRET
```

### 3. Launch TUI
```bash
chmod +x start.sh
./start.sh
```

### 4. Launch WebUI
```bash
./start.sh --web
# Open http://localhost:8080
```

### 5. Force demo mode
```bash
./start.sh --demo
```

## TUI Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `r` | Resume bot |
| `p` | Pause (no new trades) |
| `s` | Force scan now |
| `c` | Close all positions |
| `q` | Quit |

## Key Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_POSITIONS` | 5 | Max open positions at once |
| `MAX_POSITION_SIZE_PCT` | 5.0 | % of equity per trade |
| `LEVERAGE` | 5 | Leverage multiplier |
| `SL_PCT` | 0.8 | Stop loss % |
| `TP1_PCT` | 1.2 | TP1 % (after fees) |
| `TP2_PCT` | 2.0 | TP2 % |
| `TP3_PCT` | 3.5 | TP3 % |
| `SCAN_INTERVAL_SEC` | 60 | Pair scan frequency |
| `TOP_SIGNALS_N` | 10 | How many signals to use |
| `SIGNAL_EXPIRY_SEC` | 900 | Signal TTL (15 min) |
| `SIGNAL_EXPIRE_PROFIT_TAKE_PCT` | 75 | Close at X% TTL if profitable |

## Safety Features

- **Rate limiter**: stays at 100 req/min (Bybit hard limit: 120)
- **Equity guard**: never risks more than configured % per trade
- **Concurrent guard**: respects max position count
- **Breakeven SL**: after TP1, SL moves to entry+fees (can't lose)
- **Expiry guardian**: closes profitable trades before signal expires
- **Demo mode**: fully isolated from live account

## File Structure

```
h2wealth/
├── .env.example          ← copy to .env
├── start.sh              ← bootstrap + launcher
├── main.py               ← entry point
├── requirements.txt
├── core/
│   ├── config.py         ← Config, Signal, Position types
│   ├── bybit_client.py   ← Bybit v5 REST client + rate limiter
│   ├── state_store.py    ← Redis state (positions, signals, logs)
│   ├── journal.py        ← SQLite trade history
│   └── orchestrator.py   ← Main coordinator
├── signals/
│   └── engine.py         ← OFI, CVD, Funding Z, Liq, Regime
├── execution/
│   ├── risk.py           ← Position sizing, TP/SL calculation
│   └── position_manager.py ← Open, monitor, close positions
├── ui/
│   ├── webui.py          ← FastAPI dashboard + SSE
│   └── tui.py            ← Textual terminal UI
├── data/
│   └── journal.db        ← SQLite (auto-created)
└── logs/
    └── h2wealth.log      ← Log file (auto-created)
```

## Moving to Live

1. In `.env`:
   - Set `BYBIT_DEMO=false`
   - Set `BYBIT_BASE_URL=https://api.bybit.com`
   - Use your **live** API keys
2. Start small: `MAX_POSITION_SIZE_PCT=1.0`, `MAX_CONCURRENT_POSITIONS=2`
3. Monitor for several days on demo first

## Extending

- **Add a signal**: add a `calc_*` function in `signals/engine.py`, update `WEIGHTS` dict
- **Add exchange**: create `core/exchange_client.py` implementing same interface as `bybit_client.py`
- **Add pairs**: the bot auto-discovers all USDT linear perpetuals
- **Windows/macOS**: same `start.sh` works with bash; or run steps manually
