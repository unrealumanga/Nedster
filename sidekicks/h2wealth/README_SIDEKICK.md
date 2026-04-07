# ⟁ H2Wealth (Nedster Sidekick)

H2Wealth is a high-performance, automated crypto trading bot designed to work alongside Nedster. It executes trades on Bybit using non-traditional signals like Order Flow Imbalance (OFI), Cumulative Volume Delta (CVD), and Liquidity Heatmaps.

As a Nedster sidekick, Nedster has native skills to monitor, debug, and configure H2Wealth.

## Usage
1. Setup the environment:
   ```bash
   cd sidekicks/h2wealth
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Configure your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your Bybit Testnet keys
   ```
3. Run the bot:
   ```bash
   ./start.sh
   ```
