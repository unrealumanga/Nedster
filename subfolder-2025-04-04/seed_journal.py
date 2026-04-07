"""
Seed the personal library with knowledge from existing session files.
Run once: python3 seed_journal.py
"""
import os
from pathlib import Path
from journal import (capture_research, log_decision, update_project_doc,
                     save_code_snippet, JOURNAL_ROOT)

print("Seeding journal from existing session data...")

# Seed known decisions from session history
KNOWN_DECISIONS = [
    ("Never restart NH_ bots — they are quarantined", "crypto-scalper"),
    ("Use BYBIT demo account only, not live", "HYBRID"),
    ("Use python3 binary, not python", "aria-rag"),
    ("Venv always at ./venv in project directory", "aria-rag"),
    ("Aria stays fully local — no Anthropic API fallback", "aria-rag"),
    ("Prefer quarantine over delete for any project", "general"),
    ("OpenClaw and Pi both supported interfaces", "aria-rag"),
    ("num_gpu set dynamically via vram_probe.sh", "aria-rag"),
    ("BYBIT_DEMO_KEY and BYBIT_DEMO_SECRET in ~/crypto_scalper/.env", "HYBRID"),
]

for decision, project in KNOWN_DECISIONS:
    path = log_decision(decision, project=project)
    print(f"  Decision: {path.name}")

# Seed HYBRID project doc
update_project_doc(
    project="HYBRID",
    status="bootstrap phase — Bybit V5 connector in progress",
    what_works=["bootstrap.py creates directory structure",
                "settings.py loads .env keys",
                "AriaBridge loads on CPU"],
    known_issues=["No working Bybit V5 connector yet",
                  "AI signals not wired to execution",
                  "GPU OOM if second model loaded alongside aria-qwen"],
    next_steps=["Copy Bybit V5 pattern from HYDRA/",
                "Wire connector to execution loop",
                "Add RLM market session summarizer"],
    key_files={
        "~/crypto_scalper/HYBRID/main.py": "entry point",
        "~/crypto_scalper/HYBRID/src/ai/aria_bridge.py": "AI signals",
        "~/crypto_scalper/.env": "API keys (masked)",
        "~/crypto_scalper/HYDRA/test_bybit.py": "reference Bybit V5 impl",
    }
)
print("  Project: HYBRID.md seeded")

# Seed aria-rag project doc
update_project_doc(
    project="aria-rag",
    status="active — 65 fixes applied, journal system added",
    what_works=["Ollama aria-qwen model running",
                "Hybrid BM25 + vector retrieval",
                "Milestone persistence across sessions",
                "Tool execution loop (tools.py)",
                "Smart search with Tavily + DuckDuckGo fallback",
                "Think tag stripping",
                ".env masking"],
    known_issues=["num_gpu must be auto-detected at startup",
                  "Tool XML format still sometimes breaks"],
    next_steps=["Run seed_journal.py",
                "Test journal auto-capture on next session",
                "LoRA fine-tune when stable for 3 months"],
    key_files={
        "~/AI_Lab/Workspace/Nedster/rag.py": "main pipeline",
        "~/AI_Lab/Workspace/Nedster/tools.py": "tool execution",
        "~/AI_Lab/Workspace/Nedster/memory.py": "session memory",
        "~/AI_Lab/Workspace/Nedster/journal.py": "knowledge library",
        "~/AI_Lab/Workspace/Nedster/Modelfile": "Aria personality",
        "~/.aria/journal/": "persistent knowledge base",
    }
)
print("  Project: aria-rag.md seeded")

# Seed known research
capture_research(
    query="crypto trading app latest tech 2025-2026",
    findings="""Top options found:
- OpenTrader: Node.js, CCXT 100+ exchanges, paper trading, GRID/DCA/RSI strategies
- DigitalFortune: React+TypeScript, 6 Docker services, 100x leverage demo, Framer Motion
- Cryptex: Redis pub/sub, TimescaleDB, Loki+Prometheus+Grafana monitoring
- crypto-orderbook: Go backend + React frontend, real-time WebSocket, 8 exchanges
- trevortrinh/exchange: Rust CLOB + matching engine, Next.js, dual DB (PG+ClickHouse)
Recommendation: Hybrid of DigitalFortune UI + Cryptex monitoring + Rust matching engine""",
    sources=[
        "https://github.com/Open-Trader/opentrader",
        "https://github.com/abdulbaqui17/CryptoTradingPlatform",
        "https://github.com/ShivanshCharak/Cryptex",
        "https://github.com/jose-donato/crypto-orderbook",
        "https://github.com/trevor-trinh/exchange",
    ],
    topic="crypto-trading"
)
print("  Research: crypto-trading seeded")

print("\nJournal seeded. Run: python3 main.py ingest --folder ~/.aria/journal/")
