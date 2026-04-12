"""
Nedster Daemon Manager
Usage: nedster --daemon [mode1,mode2,...]
       nedster --daemon all
       nedster --daemon sentinel,trader

Each daemon is a background thread polling on its own interval.
Daemon alerts written to ~/.aria/daemon_alerts/ for pickup
at next interactive session.
"""
import threading, time, os, json, subprocess
from datetime import datetime
from pathlib import Path

ALERT_DIR = Path.home() / ".aria" / "daemon_alerts"

def write_alert(daemon: str, message: str, priority: str = "normal"):
    """Write alert for pickup at next session start."""
    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ALERT_DIR / f"{daemon}_{ts}.json"
    with open(path, "w") as f:
        json.dump({"daemon": daemon, "message": message,
                   "priority": priority,
                   "timestamp": ts}, f)

def read_pending_alerts() -> list[dict]:
    """Read all pending alerts and clear them."""
    if not ALERT_DIR.exists():
        return []
    alerts = []
    for f in ALERT_DIR.glob("*.json"):
        try:
            with open(f) as fp:
                alerts.append(json.load(fp))
            f.unlink()
        except Exception:
            pass
    return sorted(alerts, key=lambda x: x.get("timestamp",""))

class DaemonBase:
    """Base class for all Nedster daemons."""
    name = "base"
    interval_sec = 60
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self._stop = threading.Event()
        self._thread = None
    
    def check(self): raise NotImplementedError
    
    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._stop.set()
    
    def _loop(self):
        while not self._stop.is_set():
            try:
                self.check()
            except Exception as e:
                write_alert(self.name, f"Error: {e}", priority="low")
            self._stop.wait(self.interval_sec)

# ─── SENTINEL daemon ────────────────
class SentinelDaemon(DaemonBase):
    name = "sentinel"
    interval_sec = 60
    
    def check(self):
        # VRAM check
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                parts = r.stdout.strip().split(",")
                used_pct = int(parts[0]) / int(parts[1]) * 100
                if used_pct > 92:
                    write_alert("sentinel", f"⚠️ VRAM critical: {used_pct:.0f}% used. Run: ollama stop aria-qwen", priority="urgent")
        except Exception: pass
        
        # Check if known bots are still running in tmux
        bot_names = self.config.get("watch_bots", ["HYDRA", "HYBRID"])
        try:
            r = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"], capture_output=True, text=True)
            running = r.stdout.strip().split("\n")
            for bot in bot_names:
                if bot not in running:
                    write_alert("sentinel", f"Bot {bot} is not running in tmux", priority="normal")
        except Exception: pass

# ─── TRADER daemon ────────────────────────────────────────────
class TraderDaemon(DaemonBase):
    name = "trader"
    interval_sec = 300  # 5 minutes
    
    def check(self):
        import urllib.request, json
        # Quick BTC price + funding check
        try:
            url = ("https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT")
            with urllib.request.urlopen(url, timeout=6) as r:
                data = json.loads(r.read())
            t = data["result"]["list"][0]
            funding = float(t.get("fundingRate", 0)) * 100
            price = float(t["lastPrice"])
            change = float(t["price24hPcnt"]) * 100
            
            # Alert on significant moves or high funding
            if abs(change) > 3:
                write_alert("trader", f"BTC {change:+.1f}% in 24h. Price: ${price:,.0f}. Funding: {funding:+.4f}%", priority="normal")
            if abs(funding) > 0.05:
                write_alert("trader", f"⚠️ HIGH FUNDING: BTC {funding:+.4f}%/8h. Check HYDRA arb positions.", priority="urgent")
        except Exception: pass
        
        # Check bot PNL from logs
        log_base = Path.home() / "crypto_scalper"
        for bot_dir in self.config.get("bot_dirs", ["HYDRA", "HYBRID"]):
            log_path = log_base / bot_dir
            if log_path.exists():
                self._check_bot_logs(str(log_path), bot_dir)
    
    def _check_bot_logs(self, log_dir: str, bot_name: str):
        import re, glob
        logs = glob.glob(os.path.join(log_dir, "**/*.log"), recursive=True)
        if not logs: return
        newest = max(logs, key=os.path.getmtime)
        try:
            with open(newest) as f:
                lines = f.readlines()[-50:]
        except Exception: return
        
        # Look for PNL in last 50 lines
        pnl_re = re.compile(r'(?:pnl|profit|loss)[:\s]+([+-]?\d+\.?\d*)', re.I)
        total_pnl = 0.0
        for line in lines:
            m = pnl_re.search(line)
            if m: total_pnl += float(m.group(1))
        
        threshold = self.config.get("pnl_alert_threshold", -10.0)
        if total_pnl < threshold:
            write_alert("trader", f"{bot_name} PNL: {total_pnl:+.4f} USDT (threshold: {threshold})", priority="urgent")

# ─── FILE-WATCH daemon ────────────────────────────────────────
class FileWatchDaemon(DaemonBase):
    name = "file-watch"
    interval_sec = 30
    
    def check(self):
        watch_dirs = self.config.get("watch_dirs", [
            os.path.expanduser("~/AI_Lab/Workspace/Nedster"),
            os.path.expanduser("~/crypto_scalper"),
        ])
        changed = self._detect_changes(watch_dirs)
        for fpath in changed:
            self._queue_for_ingest(fpath)
    
    def _detect_changes(self, dirs: list) -> list:
        """Poll mtimes for changes in last 35s."""
        changed = []
        cutoff = time.time() - 35
        EXTS = (".py",".go",".rs",".ts",".js",".md",".yaml",".json",".sh")
        SKIP = {"venv",".venv","node_modules","__pycache__","target","chroma_db",".git"}
        for d in dirs:
            for root, dirs_list, files in os.walk(d):
                dirs_list[:] = [x for x in dirs_list if x not in SKIP]
                for f in files:
                    if not any(f.endswith(e) for e in EXTS):
                        continue
                    fpath = os.path.join(root, f)
                    try:
                        if os.path.getmtime(fpath) > cutoff:
                            changed.append(fpath)
                    except Exception: pass
        return changed
    
    def _queue_for_ingest(self, filepath: str):
        """Write to ingest queue for next session pickup."""
        queue_path = Path.home()/".aria"/"ingest_queue.json"
        try:
            queue = []
            if queue_path.exists():
                with open(queue_path) as f:
                    queue = json.load(f)
            if filepath not in queue:
                queue.append(filepath)
                with open(queue_path, "w") as f:
                    json.dump(queue[-100:], f)  # cap at 100
        except Exception: pass

# ─── MARKET daemon ───────────────────────────────────────────
class MarketDaemon(DaemonBase):
    name = "market"
    interval_sec = 14400  # 4 hours
    
    def check(self):
        import urllib.request, json
        symbols = self.config.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        lines = ["=== Market Briefing ==="]
        for sym in symbols:
            try:
                url = (f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={sym}")
                with urllib.request.urlopen(url, timeout=6) as r:
                    t = json.loads(r.read())["result"]["list"][0]
                price = float(t["lastPrice"])
                chg = float(t["price24hPcnt"]) * 100
                fund = float(t.get("fundingRate",0)) * 100
                lines.append(f"{sym}: ${price:,.2f} ({chg:+.1f}%) funding={fund:+.4f}%")
            except Exception: pass
        write_alert("market", "\n".join(lines), priority="low")

# ─── SCOUT daemon ────────────────────────────────────────────
class ScoutDaemon(DaemonBase):
    name = "scout"
    interval_sec = 14400  # 4 hours
    
    def check(self):
        """Search for AI + crypto news, save to journal."""
        topics = self.config.get("topics", [
            "ollama new release", "qwen model update",
            "bybit API changes", "local LLM benchmark 2026",
            "aria-qwen", "gguf quantization news",
        ])
        try:
            from tools import duckduckgo_search
            results = []
            for topic in topics[:3]:  # limit searches
                r = duckduckgo_search(topic)
                if r and len(r) > 30:
                    results.append(f"## {topic}\n{r[:300]}")
            
            if results:
                journal_dir = Path.home() / ".aria" / "journal" / "research"
                journal_dir.mkdir(parents=True, exist_ok=True)
                date = datetime.now().strftime("%Y-%m-%d")
                path = journal_dir / f"{date}-scout.md"
                with open(path, "w") as f:
                    f.write(f"# Scout Research {date}\n\n" + "\n\n".join(results))
                write_alert("scout", f"Research saved: {path.name}", priority="low")
        except Exception:
            pass

# ─── SKILL-BUILDER daemon ─────────────────────────────────────
class SkillBuilderDaemon(DaemonBase):
    name = "skill-builder"
    interval_sec = 604800  # weekly
    
    def check(self):
        """
        Analyze session history for gaps.
        Write SKILL.md files for recurring failure patterns.
        Inspired by Hermes agent-self-evolution.
        """
        import re
        # Read recent milestones for failure patterns
        mile_path = Path.home() / ".aria" / "milestones.md"
        if not mile_path.exists(): return
        
        with open(mile_path) as f:
            content = f.read()
        
        # Find error patterns
        ERROR_RE = re.compile(r'\[(?:FIX|ERROR|FAIL)\]\s+(.+)', re.IGNORECASE)
        errors = ERROR_RE.findall(content)
        
        if not errors: return
        
        # Group by theme
        themes = {}
        for err in errors:
            for keyword in ["git", "docker", "bybit", "ollama", "venv", "chromadb", "tool"]:
                if keyword in err.lower():
                    themes.setdefault(keyword, []).append(err)
        
        # Write skill files for top themes
        skills_dir = Path.home() / ".aria" / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        for theme, items in list(themes.items())[:3]:
            skill_path = skills_dir / f"{theme}-patterns" / "SKILL.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            with open(skill_path, "w") as f:
                f.write(f"---\nname: {theme}-patterns\ndescription: Auto-learned patterns for {theme} tasks in Nedster\n---\n# {theme.title()} Patterns\n\n## Common Issues Seen\n")
                f.write(chr(10).join(f'- {e[:80]}' for e in items[:5]))
                f.write(f"\n\n## Auto-learned Fixes\n(Populate manually or via skill-builder daemon review)\n\nGenerated: {datetime.now().strftime('%Y-%m-%d')}\n")
        write_alert("skill-builder", f"Skills updated: {list(themes.keys())[:3]}", priority="low")

# ─── Daemon registry ─────────────────────────────────────────
DAEMON_REGISTRY = {
    "sentinel":     SentinelDaemon,
    "trader":       TraderDaemon,
    "file-watch":   FileWatchDaemon,
    "market":       MarketDaemon,
    "scout":        ScoutDaemon,
    "skill-builder": SkillBuilderDaemon,
}

def start_daemon_manager(modes: list, config: dict = None) -> list:
    """Start selected daemon workers as background threads."""
    config = config or {}
    active = []
    for mode in modes:
        cls = DAEMON_REGISTRY.get(mode)
        if cls:
            d = cls(config.get(mode, {}))
            d.start()
            active.append(d)
            print(f"[Daemon] {mode} started (interval: {d.interval_sec}s)")
        else:
            print(f"[Daemon] Unknown mode: {mode}")
    return active
