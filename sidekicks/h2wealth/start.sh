#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  H2Wealth Trading Bot — Bootstrap & Launch Script
#  Usage:
#    ./start.sh              → TUI (default)
#    ./start.sh --web        → WebUI on port 8080
#    ./start.sh --demo       → force demo mode
#    ./start.sh --check-only → run checklist and exit
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()   { echo -e "${CYAN}[H2W]${NC} $*"; }
ok()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[⚠]${NC} $*"; }
fail()  { echo -e "${RED}[✗]${NC} $*"; }
die()   { fail "$*"; exit 1; }

echo -e "${BOLD}${BLUE}"
echo " ██╗  ██╗██████╗ ██╗    ██╗███████╗ █████╗ ██╗  ████████╗██╗  ██╗"
echo " ██║  ██║╚════██╗██║    ██║██╔════╝██╔══██╗██║  ╚══██╔══╝██║  ██║"
echo " ███████║ █████╔╝██║ █╗ ██║█████╗  ███████║██║     ██║   ███████║"
echo " ██╔══██║██╔═══╝ ██║███╗██║██╔══╝  ██╔══██║██║     ██║   ██╔══██║"
echo " ██║  ██║███████╗╚███╔███╔╝███████╗██║  ██║███████╗██║   ██║  ██║"
echo " ╚═╝  ╚═╝╚══════╝ ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝   ╚═╝  ╚═╝"
echo -e "${NC}"
echo -e "${BOLD}  Automated Crypto Trading Bot — Bybit v5${NC}"
echo ""

# ── Parse args ──────────────────────────────────────────────────────
MODE="tui"
CHECK_ONLY=false
FORCE_DEMO=false
for arg in "$@"; do
  case "$arg" in
    --web)        MODE="web";;
    --demo)       FORCE_DEMO=true;;
    --check-only) CHECK_ONLY=true;;
  esac
done

# ════════════════════════════════════════════════════════════════════
# STEP 1: System dependencies
# ════════════════════════════════════════════════════════════════════
log "Checking system requirements..."

if ! command -v python3 &>/dev/null; then
  die "Python 3 not found. Install: sudo apt install python3 python3-pip python3-venv"
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo $PY_VER | cut -d. -f1)
PY_MINOR=$(echo $PY_VER | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
  die "Python 3.10+ required (found $PY_VER)"
fi
ok "Python $PY_VER"

if ! command -v redis-cli &>/dev/null; then
  warn "redis-cli not found. Attempting install..."
  if command -v apt-get &>/dev/null; then
    sudo apt-get install -y redis-server redis-tools 2>/dev/null || die "Redis install failed"
  elif command -v brew &>/dev/null; then
    brew install redis 2>/dev/null || die "Redis install failed"
  else
    die "Cannot install Redis automatically. Please install: https://redis.io"
  fi
fi
ok "Redis available"

if ! redis-cli ping &>/dev/null 2>&1; then
  log "Starting Redis server..."
  if command -v systemctl &>/dev/null; then
    sudo systemctl start redis-server 2>/dev/null || redis-server --daemonize yes --logfile /tmp/redis-h2w.log
  else
    redis-server --daemonize yes --logfile /tmp/redis-h2w.log
  fi
  sleep 1
fi
if redis-cli ping &>/dev/null 2>&1; then
  ok "Redis running ($(redis-cli ping))"
else
  die "Redis failed to start"
fi

# ════════════════════════════════════════════════════════════════════
# STEP 2: Python virtual environment
# ════════════════════════════════════════════════════════════════════
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
  log "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
  ok "venv created at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
ok "Virtual environment activated"

log "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
ok "Dependencies installed"

# ════════════════════════════════════════════════════════════════════
# STEP 3: Environment file
# ════════════════════════════════════════════════════════════════════
log "Checking .env configuration..."

if [ ! -f "$SCRIPT_DIR/.env" ]; then
  if [ -f "$SCRIPT_DIR/.env.example" ]; then
    warn ".env not found. Copying .env.example → .env"
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  ACTION REQUIRED: Edit .env and set your Bybit API keys:${NC}"
    echo -e "${YELLOW}    BYBIT_API_KEY=...${NC}"
    echo -e "${YELLOW}    BYBIT_API_SECRET=...${NC}"
    echo -e "${YELLOW}  Then run ./start.sh again${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
  else
    die ".env and .env.example both missing"
  fi
fi

set -a; source "$SCRIPT_DIR/.env"; set +a

if [ "$FORCE_DEMO" = "true" ]; then
  export BYBIT_DEMO=true
  export BYBIT_BASE_URL="https://api-demo.bybit.com"
fi

# ════════════════════════════════════════════════════════════════════
# STEP 4: API Key Validation
# ════════════════════════════════════════════════════════════════════
log "Validating API keys..."

if [ -z "${BYBIT_API_KEY:-}" ] || [ "${BYBIT_API_KEY}" = "your_api_key_here" ]; then
  die "BYBIT_API_KEY not set in .env"
fi
if [ -z "${BYBIT_API_SECRET:-}" ] || [ "${BYBIT_API_SECRET}" = "your_api_secret_here" ]; then
  die "BYBIT_API_SECRET not set in .env"
fi
ok "API keys present"

# ════════════════════════════════════════════════════════════════════
# STEP 5: Connectivity Checklist
# ════════════════════════════════════════════════════════════════════
log "Running connectivity checklist..."

# Internet check against the configured base URL host
BASE_HOST=$(echo "${BYBIT_BASE_URL:-https://api-demo.bybit.com}" | sed 's|https://||' | cut -d/ -f1)
if curl -s --max-time 5 "https://${BASE_HOST}/v5/market/time" -o /dev/null; then
  ok "Internet → ${BASE_HOST}: reachable"
else
  warn "Cannot reach ${BASE_HOST} — check network / VPN"
fi

# ── Bybit API ping (authenticated) ──────────────────────────────────
# Write a self-contained script to a temp file to avoid subshell print pollution
PING_SCRIPT=$(mktemp /tmp/h2w_ping_XXXXXX.py)
cat > "$PING_SCRIPT" << 'PYEOF'
import asyncio, sys, os
sys.path.insert(0, os.environ.get("H2W_DIR", "."))
from core.config import Config
from core.bybit_client import BybitClient

async def main():
    cfg = Config()
    c = BybitClient(cfg)
    await c.start()
    result = await c.ping()
    await c.stop()
    # Print ONLY the result token — nothing else
    sys.stdout.write("ok" if result else "fail")
    sys.stdout.flush()

asyncio.run(main())
PYEOF

export H2W_DIR="$SCRIPT_DIR"
PING_RESULT=$(python3 "$PING_SCRIPT" 2>/dev/null)
rm -f "$PING_SCRIPT"

DEMO_TAG=""
[ "${BYBIT_DEMO:-true}" = "true" ] && DEMO_TAG=" [DEMO — api-demo.bybit.com]"

if [ "$PING_RESULT" = "ok" ]; then
  ok "Bybit API${DEMO_TAG}: connected and keys valid"
else
  fail "Bybit API${DEMO_TAG}: ping failed (got: '${PING_RESULT}')"
  echo ""
  warn "Possible causes:"
  echo "  1. Invalid API keys — regenerate on Bybit dashboard"
  echo "  2. API key missing 'Contract Trading' permission"
  echo "  3. IP not whitelisted (check Bybit API key settings)"
  echo "  4. Wrong URL — current: ${BYBIT_BASE_URL:-not set}"
  echo ""
  echo -e "  ${CYAN}Bybit Demo URL:${NC} https://api-demo.bybit.com"
  echo -e "  ${CYAN}Bybit Live URL:${NC} https://api.bybit.com"
  echo -e "  ${CYAN}Bybit Testnet:${NC}  https://api-testnet.bybit.com  (different from demo!)"
  echo ""
  read -p "Continue anyway? (y/N): " CONT
  [[ "$CONT" != "y" && "$CONT" != "Y" ]] && exit 1
fi

# Redis
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping &>/dev/null; then
  ok "Redis at $REDIS_HOST:$REDIS_PORT: connected"
else
  die "Redis at $REDIS_HOST:$REDIS_PORT: unreachable"
fi

mkdir -p logs data
ok "Directories: logs/ data/ ready"

# ════════════════════════════════════════════════════════════════════
# STEP 6: Config summary
# ════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  Configuration Summary${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Mode:              $([ "${BYBIT_DEMO:-true}" = "true" ] && echo "${YELLOW}DEMO${NC}" || echo "${RED}LIVE${NC}")"
echo -e "  API URL:           ${BYBIT_BASE_URL:-https://api-demo.bybit.com}"
echo -e "  Max positions:     ${MAX_CONCURRENT_POSITIONS:-5}"
echo -e "  Position size:     ${MAX_POSITION_SIZE_PCT:-5}% of equity per trade"
echo -e "  Leverage:          ${LEVERAGE:-5}x"
echo -e "  SL:                ${SL_PCT:-0.8}%"
echo -e "  TP1/TP2/TP3:       ${TP1_PCT:-1.2}% / ${TP2_PCT:-2.0}% / ${TP3_PCT:-3.5}%"
echo -e "  Scan interval:     ${SCAN_INTERVAL_SEC:-60}s"
echo -e "  Top signals N:     ${TOP_SIGNALS_N:-10}"
echo -e "  Signal TTL:        ${SIGNAL_EXPIRY_SEC:-900}s"
echo -e "  UI mode:           $MODE"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ "$CHECK_ONLY" = "true" ]; then
  ok "Checklist complete. Exiting (--check-only)."
  exit 0
fi

# ════════════════════════════════════════════════════════════════════
# STEP 7: Launch
# ════════════════════════════════════════════════════════════════════
log "Launching H2Wealth..."
echo ""

if [ "$MODE" = "web" ]; then
  echo -e "  ${GREEN}WebUI starting at http://localhost:${WEBUI_PORT:-8080}${NC}"
  echo ""
  exec python3 main.py --web "$@"
else
  exec python3 main.py "$@"
fi
