"""
H2Wealth - Entry Point
Usage:
  python main.py          → TUI (default)
  python main.py --web    → WebUI on port 8080
  python main.py --demo   → run with demo account (override .env)
"""
import asyncio, logging, os, sys
from pathlib import Path

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

from core.config import Config

cfg = Config()

# Configure logging
logging.basicConfig(
    level   = getattr(logging, cfg.log_level.upper(), logging.INFO),
    format  = "%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers= [
        logging.FileHandler(cfg.log_file),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("main")


def main():
    args = sys.argv[1:]

    if "--demo" in args:
        os.environ["BYBIT_DEMO"] = "true"
        os.environ["BYBIT_BASE_URL"] = "https://api-demo.bybit.com"
        log.info("DEMO mode forced via CLI")

    if "--web" in args:
        log.info(f"Starting WebUI on {cfg.webui_host}:{cfg.webui_port}")
        from ui.webui import run_webui
        run_webui(cfg)
    else:
        log.info("Starting TUI")
        from ui.tui import run_tui
        asyncio.run(run_tui())


if __name__ == "__main__":
    main()
