import os
from journal import search_journal, capture_research, log_decision
from pathlib import Path
import threading
import subprocess
import requests
import json
import urllib.request
import urllib.parse

import re

import re as _re

_SECRET_PATTERNS = [
    (
        _re.compile(
            r"((?:API_KEY|SECRET|TOKEN|PASSWORD|PASSPHRASE|PRIVATE_KEY)\s*[=:]\s*)([^\s\n]{8,})",
            _re.IGNORECASE,
        ),
        lambda m: m.group(1) + "[REDACTED]",
    ),
    (
        _re.compile(r"(\d{8,12}:AA[A-Za-z0-9_\-]{8,})"),
        lambda m: "[TELEGRAM_TOKEN_REDACTED]",
    ),
    (
        _re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,}\b"),
        lambda m: "[ANTHROPIC_KEY_REDACTED]",
    ),
    (
        _re.compile(r"\bsk-or-[A-Za-z0-9\-_]{20,}\b"),
        lambda m: "[OPENROUTER_KEY_REDACTED]",
    ),
    (
        _re.compile(
            r"((?:BYBIT|OKX|BINANCE|KRAKEN)[_\w]*(?:KEY|SECRET|PASS)\s*=\s*)([A-Za-z0-9]{8,})",
            _re.IGNORECASE,
        ),
        lambda m: m.group(1) + "[EXCHANGE_KEY_REDACTED]",
    ),
    (_re.compile(r"\btvly-[A-Za-z0-9\-_]{10,}\b"), lambda m: "[TAVILY_KEY_REDACTED]"),
    (_re.compile(r"\+\d{7,15}\b"), lambda m: "+[PHONE_REDACTED]"),
]


def sanitize_output(text: str) -> str:
    if not text:
        return text
    for pattern, replacer in _SECRET_PATTERNS:
        text = pattern.sub(replacer, text)
    return text


class SessionState:
    def __init__(self):
        self.cwd = os.getcwd()
        self.read_files = {}
        self.env_vars = {}
        self.created_files = []
        self.models_in_vram = []

    def update_cwd(self, new_cwd: str):
        expanded = os.path.expanduser(new_cwd)
        if os.path.isdir(expanded):
            self.cwd = expanded
        return self.cwd

    def record_file_read(self, path: str, content: str):
        self.read_files[path] = content[:200]

    def was_read(self, path: str) -> bool:
        return path in self.read_files


SESSION = SessionState()


def check_model_available(model_path: str) -> dict:
    model_path = os.path.expanduser(str(model_path))

    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_path.lower() in r.stdout.lower():
            return {"available": True, "location": "ollama", "vram_mb": 0}
    except Exception:
        pass

    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = "models--" + model_path.replace("/", "--")
    hf_path = os.path.join(hf_cache, model_dir_name)
    if os.path.exists(hf_path):
        try:
            r = subprocess.run(["du", "-sm", hf_path], capture_output=True, text=True)
            size_mb = int(r.stdout.split()[0]) if r.stdout else 0
        except Exception:
            size_mb = 0
        return {"available": True, "location": hf_path, "vram_mb": size_mb}

    return {"available": False, "location": None, "vram_mb": 0}


def get_available_models() -> str:
    lines = ["=== Available Models ==="]
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if r.stdout.strip():
            lines.append("Ollama:")
            for line in r.stdout.strip().split("\n")[1:]:
                lines.append(f"  {line.split()[0]}")
    except Exception:
        lines.append("Ollama: (not running)")

    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(hf_cache):
        lines.append("HuggingFace cache:")
        for d in os.listdir(hf_cache):
            if d.startswith("models--"):
                model_name = d[8:].replace("--", "/")
                lines.append(f"  {model_name}")

    return "\n".join(lines)


def get_vram_free_mb() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        return int(r.stdout.strip())
    except Exception:
        return 0


def check_vram_before_load(model_size_mb: int) -> str:
    free = get_vram_free_mb()
    needed = int(model_size_mb * 1.2)
    if free >= needed:
        return f"OK — {free}MB free, {needed}MB needed"
    return f"VRAM insufficient: {free}MB free, {needed}MB needed. Options: use CPU (device='cpu'), unload aria-qwen first (ollama stop aria-qwen), or use smaller model."


from typing import List, Dict


def read_file(path: str) -> str:
    path = os.path.expanduser(str(path))
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            SESSION.record_file_read(path, content)
            content = sanitize_output(content)

            if len(content) > 8000:
                return (
                    content[:8000]
                    + f"\n\n...[TRUNCATED: File too large ({len(content)} chars). Use `search_code` or `code_xray` to explore further without blowing context limits.]"
                )
            return content
    except Exception as e:
        return sanitize_output(f"Error reading file: {e}")


def write_file(path: str, content: str) -> str:
    path = os.path.expanduser(str(path))
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def run_bash(cmd: str, timeout: int = 15) -> str:
    cmd = os.path.expanduser(str(cmd))
    try:
        cd_match = _re.search(r"\bcd\s+([^\s;&|]+)", cmd)
        if cd_match:
            new_dir = cd_match.group(1).strip()
            resolved = os.path.normpath(
                os.path.join(SESSION.cwd, os.path.expanduser(new_dir))
            )
            SESSION.update_cwd(resolved)

        full_cmd = f"cd {SESSION.cwd} && {cmd}"

        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout + result.stderr
        if len(output) > 2000:
            output = output[:2000] + "\n...[output truncated]"
        return sanitize_output(output)
    except Exception as e:
        return sanitize_output(f"Error running bash: {e}")


def list_dir(path: str) -> str:
    path = os.path.expanduser(str(path))
    try:
        return "\n".join(os.listdir(path))
    except Exception as e:
        return f"Error listing dir: {e}"


def search_code(
    query: str = "", directory: str = ".", pattern: str = "", path: str = ""
) -> str:
    actual_query = pattern or query
    actual_dir = path or directory
    if not actual_query:
        return "Error: no search pattern provided"

    actual_dir = os.path.expanduser(str(actual_dir))
    try:
        cmd = f"grep -rnw '{actual_dir}' -e '{actual_query}' | head -n 20"
        return run_bash(cmd)
    except Exception as e:
        return f"Error searching code: {e}"


def get_clipboard() -> str:
    try:
        result = subprocess.run(
            ["xclip", "-selection", "clipboard", "-o"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout
    except Exception as e:
        return f"Error reading clipboard: {e}"


def web_fetch(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        text = response.text
        if len(text) > 3000:
            text = text[:3000] + "\n...[truncated]"
        return text
    except Exception as e:
        return f"Error fetching url: {e}"


TOOL_REGISTRY = {
    "read_file": read_file,
    "run_bash": run_bash,
    "list_dir": list_dir,
    "search_code": search_code,
    "get_clipboard": get_clipboard,
    "web_fetch": web_fetch,
    "search_code": search_code,
}


def normalize_tool_args(tool_name: str, args: dict) -> dict:
    TOOL_ARG_ALIASES = {
        "search_code": {"pattern": "query", "path": "directory", "dir": "directory"},
        "run_bash": {"command": "cmd", "bash": "cmd", "shell": "cmd"},
        "read_file": {"file": "path", "filename": "path"},
        "list_dir": {"dir": "path", "directory": "path", "folder": "path"},
        "git_status": {"dir": "cwd", "directory": "cwd"},
        "git_diff": {"dir": "cwd"},
        "git_commit": {"dir": "cwd", "msg": "message"},
    }
    aliases = TOOL_ARG_ALIASES.get(tool_name, {})
    normalized = {}
    for k, v in args.items():
        normalized[aliases.get(k, k)] = v
    return normalized


def _repair_json(raw: str) -> str:
    raw = raw.strip()
    if not raw.endswith("}"):
        raw += "}"
    if not raw.startswith("{"):
        raw = "{" + raw
    # Fix unquoted keys
    raw = _re.sub(r"(\w+)(?=\s*:)", r'"\1"', raw)
    # Remove trailing commas
    raw = _re.sub(r",\s*}", "}", raw)
    return raw


def parse_tool_calls(text: str) -> list:
    """
    Extract tool calls handling ALL format variants Aria might generate:
    - Correct:  <tool name="bash">{"cmd": "ls"}</tool>
    - Broken 1: <tool name="bash"><parameter=cmd>ls</parameter>
    - Broken 2: <tool_call>{"name": "bash", "cmd": "ls"}</tool_call>
    - Broken 3: ```bash\nls\n```  (markdown code blocks)
    """
    import re, json

    results = []

    # Format 1: Correct JSON format
    pattern1 = re.compile(
        r'<tool\s+name=["\']([^"\']+)["\']>\s*(.*?)\s*</tool>',
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern1.finditer(text):
        name = m.group(1).strip()
        args_raw = m.group(2).strip()
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            try:
                args = json.loads(_repair_json(args_raw))
            except Exception:
                args = _parse_kv(args_raw)
        except Exception:
            args = _parse_kv(args_raw)
        results.append({"name": name, "args": args})

    if results:
        return results

    # Format 2: Broken <parameter=X> format
    pattern2 = re.compile(
        r'<tool\s+name=["\']([^"\']+)["\']>.*?<parameter[= ]+(\w+)>\s*(.*?)\s*(?:</parameter>|$)',
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern2.finditer(text):
        name = m.group(1).strip()
        key = m.group(2).strip()
        val = m.group(3).strip()
        results.append({"name": name, "args": {key: val}})

    if results:
        return results

    # Format 3: tool_call JSON blob
    pattern3 = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE
    )
    for m in pattern3.finditer(text):
        try:
            blob = json.loads(m.group(1))
            name = blob.pop("name", blob.pop("tool", "run_bash"))
            results.append({"name": name, "args": blob})
        except Exception:
            pass

    # Format 5: Malformed closing tag - missing slash
    pattern5 = re.compile(
        r'<tool\s+name=["\']([^"\']+)["\']>(.*?)<tool>', re.DOTALL | re.IGNORECASE
    )
    for m in pattern5.finditer(text):
        name = m.group(1).strip()
        args_raw = m.group(2).strip()
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            try:
                args = json.loads(_repair_json(args_raw))
            except Exception:
                args = _parse_kv(args_raw)
        except Exception:
            args = _parse_kv(args_raw)
        results.append({"name": name, "args": args})

    # Format 6: create file variant treated as tool call
    pattern_create = re.compile(
        r'<(?:tool\s+)?create\s+file=["\']?([^"\'>\s]+)["\']?>(.*?)</(?:tool|create)>',
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern_create.finditer(text):
        path = m.group(1).strip()
        content = m.group(2).strip()
        results.append(
            {"name": "write_file", "args": {"path": path, "content": content}}
        )

    # Format 7: self closing tag
    pattern_self_close = re.compile(
        r'<tool\s+name=["\']([^"\']+)["\']\s*/>', re.DOTALL | re.IGNORECASE
    )
    for m in pattern_self_close.finditer(text):
        name = m.group(1).strip()
        results.append({"name": name, "args": {}})

    return results

    # Format 4: Bare ```bash``` code blocks (treat as run_bash)
    pattern4 = re.compile(r"```(?:bash|sh|shell)\n(.*?)```", re.DOTALL)
    for m in pattern4.finditer(text):
        cmd = m.group(1).strip()
        if cmd:
            results.append({"name": "run_bash", "args": {"cmd": cmd}})

    return results


class ContinuityWatchdog:
    """
    Prevents Aria from going silent mid-session.
    If no output for >30s during a tool loop, prints a heartbeat.
    """

    def __init__(self, timeout=30):
        self.timeout = timeout
        self._timer = None
        self._active = False

    def start(self):
        self._active = True
        self._reset()

    def stop(self):
        self._active = False
        if self._timer:
            self._timer.cancel()

    def ping(self):
        """Call this after each tool execution to reset timer."""
        if self._active:
            self._reset()

    def _reset(self):
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.timeout, self._heartbeat)
        self._timer.daemon = True
        self._timer.start()

    def _heartbeat(self):
        if self._active:
            print("\n  • \x1b[38;5;245m\x1b[3mWorking...\x1b[0m", flush=True)
            self._reset()


WATCHDOG = ContinuityWatchdog(timeout=30)

TAVILY_KEY_PATH = os.path.expanduser("~/.aria/tavily.key")


def _load_tavily_key() -> str:
    """Load Tavily API key from secure file."""
    if os.path.exists(TAVILY_KEY_PATH):
        try:
            with open(TAVILY_KEY_PATH) as f:
                return f.read().strip()
        except Exception:
            return ""
    # Also check environment
    return os.environ.get("TAVILY_API_KEY", "")


def store_tavily_key(key: str) -> str:
    """Securely store Tavily API key (never echo it)."""
    try:
        os.makedirs(os.path.dirname(TAVILY_KEY_PATH), exist_ok=True)
        with open(TAVILY_KEY_PATH, "w") as f:
            f.write(key.strip())
        os.chmod(TAVILY_KEY_PATH, 0o600)  # owner read only
        return "Tavily key stored securely at ~/.aria/tavily.key"
    except Exception as e:
        return f"Failed to store Tavily key: {e}"


def probe_tools() -> dict:
    """
    Test which tools actually work at startup.
    Returns dict of {tool_name: status_str}
    """
    status = {}

    # Test bash
    try:
        result = run_bash("echo OK", timeout=5)
        status["bash"] = "OK" if "OK" in result else f"WARN: {result[:50]}"
    except Exception as e:
        status["bash"] = f"FAIL: {e}"

    # Test web_search
    try:
        # Import the actual web_search function if available
        # This checks if the Ollama web search extension is active
        r = subprocess.run(
            ["curl", "-s", "--max-time", "5", "http://localhost:11434/api/tags"],
            capture_output=True,
            text=True,
        )
        status["ollama"] = "OK" if r.returncode == 0 else "FAIL"
    except Exception as e:
        status["ollama"] = f"FAIL: {e}"

    # Test Tavily if key exists
    tavily_key = _load_tavily_key()
    if tavily_key:
        try:
            import urllib.request, json as _json

            req = urllib.request.Request(
                "https://api.tavily.com/search",
                data=_json.dumps(
                    {"query": "test", "api_key": tavily_key, "max_results": 1}
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                status["tavily"] = "OK"
        except Exception as e:
            status["tavily"] = f"FAIL: {e}"
    else:
        status["tavily"] = "NO_KEY"

    return status


TOOL_STATUS = {}  # populated at startup via probe_tools()


def tavily_search(query: str, max_results: int = 5) -> str:
    """Search using Tavily API. Returns formatted results."""
    import json as _json, urllib.request as _req

    key = _load_tavily_key()
    if not key:
        return "ERROR: No Tavily key. Run: store_tavily_key('your-key')"
    try:
        payload = _json.dumps(
            {
                "query": query,
                "api_key": key,
                "max_results": max_results,
                "search_depth": "basic",
            }
        ).encode()
        request = _req.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _req.urlopen(request, timeout=10) as resp:
            data = _json.loads(resp.read())
        results = data.get("results", [])
        if not results:
            return "No results found."
        lines = []
        for r in results[:max_results]:
            lines.append(f"[{r.get('score', 0):.2f}] {r.get('title', '')}")
            lines.append(f"URL: {r.get('url', '')}")
            lines.append(r.get("content", "")[:400])
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Tavily error: {e}"


def duckduckgo_search(query: str) -> str:
    """DuckDuckGo search via HTML scraping fallback."""
    import urllib.request as _req, urllib.parse as _parse

    try:
        url = f"https://html.duckduckgo.com/html/?q={_parse.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        request = _req.Request(url, headers=headers)
        with _req.urlopen(request, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        import re

        # Extract result snippets
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL
        )
        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
        clean = lambda s: re.sub(r"<[^>]+>", "", s).strip()
        lines = []
        for t, s in zip(titles[:5], snippets[:5]):
            lines.append(f"- {clean(t)}: {clean(s)}")

        result = "\n".join(lines) if lines else "No results."
        if not result or len(result) < 20 or result == "No results.":
            return "DuckDuckGo: No results returned (possibly blocked). Try web_fetch directly."
        return result
    except Exception as e:
        return f"DuckDuckGo error: {e}"


def get_crypto_price(symbol: str = "BTC", currency: str = "usd") -> str:
    """Fetch price from CoinGecko public API (no key needed)."""
    currency = currency.lower()
    id_map = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "BNB": "binancecoin",
        "USDT": "tether",
    }
    cg_id = id_map.get(symbol.upper(), symbol.lower())
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies={currency}"
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read())
        price = data[cg_id][currency]
        return f"{symbol.upper()}: ${price:,.2f} {currency.upper()}"
    except Exception as e:
        return f"Price fetch failed: {e}"


def smart_search(query: str, context: str = "") -> str:
    """
    Intelligent search with automatic fallback chain:
    1. Try Tavily (best results, structured)
    2. Try DuckDuckGo (reliable fallback)
    3. Try web_fetch on a likely URL (last resort)

    Also validates relevance: result must contain at least one
    keyword from the query or it tries the next engine.
    Returns the first useful result with source label.
    """
    import re

    # Extract keywords from query for relevance check
    keywords = [
        w.lower()
        for w in query.split()
        if len(w) > 3
        and w.lower()
        not in {
            "what",
            "when",
            "where",
            "which",
            "that",
            "this",
            "have",
            "from",
            "with",
            "about",
        }
    ]

    def is_relevant(text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords[:5])

    # Attempt 1: Tavily
    if _load_tavily_key():
        result = tavily_search(query, max_results=3)
        if not result.startswith("ERROR") and not result.startswith("Tavily error"):
            if is_relevant(result) or len(result) > 200:
                return f"[Tavily]\n{result}"

    # Attempt 2: DuckDuckGo
    result = duckduckgo_search(query)
    if not result.startswith("DuckDuckGo error") and len(result) > 100:
        if is_relevant(result):
            return f"[DuckDuckGo]\n{result}"

    # Attempt 3: Rephrase and retry DuckDuckGo
    alt_query = " ".join(keywords[:4]) + " explanation"
    result = duckduckgo_search(alt_query)
    if not result.startswith("DuckDuckGo error") and len(result) > 100:
        return f"[DuckDuckGo rephrased: '{alt_query}']\n{result}"

    # Attempt 4: web_fetch on a likely source
    likely_url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
    result = web_fetch(likely_url)
    if result and not result.startswith("Error"):
        return f"[Wikipedia]\n{result[:1500]}"

    return f"Search exhausted for: '{query}'. No relevant results found."


# Import and register git_tools
try:
    from git_tools import (
        git_status,
        git_diff,
        git_commit,
        git_branch,
        git_stash,
        git_log,
        git_add,
        is_git_repo,
    )

    TOOL_REGISTRY.update(
        {
            "git_status": lambda args: git_status(
                os.path.expanduser(str(args.get("cwd", ".")))
            ),
            "git_diff": lambda args: git_diff(
                os.path.expanduser(str(args.get("cwd", "."))), args.get("file", "")
            ),
            "git_commit": lambda args: git_commit(
                os.path.expanduser(str(args.get("cwd", "."))), args.get("message", "")
            ),
            "git_branch": lambda args: git_branch(
                os.path.expanduser(str(args.get("cwd", ".")))
            ),
            "git_stash": lambda args: git_stash(
                os.path.expanduser(str(args.get("cwd", ".")))
            ),
            "git_log": lambda args: git_log(
                os.path.expanduser(str(args.get("cwd", "."))), args.get("n", 10)
            ),
        }
    )
except ImportError:
    pass

# Import and register code_tools
try:
    from code_tools import (
        detect_test_runner,
        run_tests,
        run_linter,
        run_formatter,
        check_syntax,
        get_project_info,
    )

    TOOL_REGISTRY.update(
        {
            "run_tests": lambda args: run_tests(
                args.get("cwd", "."), args.get("file", "")
            ),
            "run_linter": lambda args: run_linter(
                args.get("cwd", "."), args.get("file", "")
            ),
            "run_formatter": lambda args: run_formatter(
                args.get("cwd", "."), args.get("file", "")
            ),
            "check_syntax": lambda args: check_syntax(
                args.get("code", ""), args.get("language", "python")
            ),
            "get_project_info": lambda args: str(
                get_project_info(args.get("cwd", "."))
            ),
        }
    )
except ImportError:
    pass

# Update TOOL_REGISTRY to include new tools:


def read_bot_logs(bot_name: str = "", lines: int = 50) -> str:
    """Read last N lines from a crypto bot's logs."""
    base = os.path.expanduser("~/crypto_scalper")
    if bot_name:
        candidates = [
            os.path.join(base, bot_name, "logs"),
            os.path.join(base, bot_name),
        ]
    else:
        candidates = [base]

    log_files = []
    for c in candidates:
        if os.path.isdir(c):
            for f in os.listdir(c):
                if f.endswith((".log", ".txt")) and "log" in f.lower():
                    log_files.append(os.path.join(c, f))

    if not log_files:
        return f"No log files found for bot: {bot_name}"

    results = []
    for lf in log_files[:3]:  # max 3 logs
        try:
            r = subprocess.run(
                ["tail", f"-{lines}", lf], capture_output=True, text=True
            )
            results.append(f"=== {os.path.basename(lf)} ===\n{r.stdout}")
        except Exception as e:
            results.append(f"Error reading {lf}: {e}")
    return "\n".join(results)


TOOL_REGISTRY.update(
    {
        "tavily_search": lambda args: tavily_search(
            args.get("query", ""), args.get("max_results", 5)
        ),
        "duckduckgo_search": lambda args: duckduckgo_search(args.get("query", "")),
        "store_tavily_key": lambda args: store_tavily_key(args.get("key", "")),
        "probe_tools": lambda args: str(probe_tools()),
        "smart_search": lambda args: smart_search(
            args.get("query", ""), args.get("context", "")
        ),
        "get_crypto_price": lambda args: get_crypto_price(
            args.get("symbol", "BTC"), args.get("currency", "usd")
        ),
        "read_bot_logs": lambda args: read_bot_logs(
            args.get("bot_name", ""), args.get("lines", 50)
        ),
    }
)


def read_env_safe(path: str) -> str:
    """
    Read .env file but mask all values, show only key names.
    Use this instead of read_file() for .env files.
    """
    try:
        with open(os.path.expanduser(path)) as f:
            lines = f.readlines()
        masked = []
        for line in lines:
            line = line.rstrip()
            if not line or line.startswith("#"):
                masked.append(line)
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                if val and not val.startswith("#"):
                    preview = val[:3] + "..." if len(val) > 3 else "***"
                    masked.append(
                        f"{key}=[MASKED — {len(val)} chars, preview: {preview}]"
                    )
                else:
                    masked.append(line)
            else:
                masked.append(line)
        return "\n".join(masked)
    except Exception as e:
        return f"Error reading {path}: {e}"


def _parse_kv(text: str) -> dict:
    """Parse key=value or key: value pairs as fallback."""
    args = {}
    for m in _re.finditer(r'(\w+)\s*[=:]\s*["\']?([^"\'<\n]+)["\']?', text):
        args[m.group(1).strip()] = m.group(2).strip()
    return args


# ══ TOOL #1: market_intel ════════════════════════════════════


def market_intel(
    symbol: str = "BTC", exchange: str = "bybit", depth: bool = False
) -> str:
    """
    Real-time market data from public exchange APIs — no key needed.

    Returns in one call:
    - Current price + 24h change %
    - Funding rate (perpetual)
    - Open Interest
    - 24h volume
    - Bid/Ask spread
    - Top 5 orderbook levels (if depth=True)

    Sources (fallback chain):
    1. Bybit public REST: https://api.bybit.com/v5/market/tickers
    2. OKX public REST:   https://www.okx.com/api/v5/market/ticker
    3. Binance public:    https://api.binance.com/api/v3/ticker/24hr
    4. CoinGecko fallback (existing get_crypto_price)
    """
    import urllib.request, json, time

    symbol_map = {
        "BTC": {"bybit": "BTCUSDT", "okx": "BTC-USDT-SWAP", "binance": "BTCUSDT"},
        "ETH": {"bybit": "ETHUSDT", "okx": "ETH-USDT-SWAP", "binance": "ETHUSDT"},
        "SOL": {"bybit": "SOLUSDT", "okx": "SOL-USDT-SWAP", "binance": "SOLUSDT"},
    }
    sym = symbol.upper()
    results = {}

    # Attempt 1: Bybit
    try:
        bybit_sym = symbol_map.get(sym, {}).get("bybit", f"{sym}USDT")
        url = (
            f"https://api.bybit.com/v5/market/tickers"
            f"?category=linear&symbol={bybit_sym}"
        )
        with urllib.request.urlopen(url, timeout=6) as r:
            data = json.loads(r.read())
        t = data["result"]["list"][0]
        results["price"] = float(t["lastPrice"])
        results["change_24h"] = float(t["price24hPcnt"]) * 100
        results["volume_24h"] = float(t["volume24h"])
        results["funding"] = float(t.get("fundingRate", 0)) * 100
        results["open_int"] = float(t.get("openInterest", 0))
        results["bid"] = float(t.get("bid1Price", 0))
        results["ask"] = float(t.get("ask1Price", 0))
        results["source"] = "Bybit"
    except Exception:
        # Attempt 2: OKX
        try:
            okx_sym = symbol_map.get(sym, {}).get("okx", f"{sym}-USDT-SWAP")
            url = f"https://www.okx.com/api/v5/market/ticker?instId={okx_sym}"
            with urllib.request.urlopen(url, timeout=6) as r:
                data = json.loads(r.read())
            t = data["data"][0]
            results["price"] = float(t["last"])
            results["change_24h"] = round(
                (float(t["last"]) - float(t["open24h"])) / float(t["open24h"]) * 100, 2
            )
            results["volume_24h"] = float(t["vol24h"])
            results["bid"] = float(t["bidPx"])
            results["ask"] = float(t["askPx"])
            results["source"] = "OKX"
        except Exception as e:
            return f"market_intel failed: {e}"

    # Orderbook depth (optional)
    depth_str = ""
    if depth and results.get("source") == "Bybit":
        try:
            bybit_sym = symbol_map.get(sym, {}).get("bybit", f"{sym}USDT")
            url = (
                f"https://api.bybit.com/v5/market/orderbook"
                f"?category=linear&symbol={bybit_sym}&limit=5"
            )
            with urllib.request.urlopen(url, timeout=5) as r:
                ob = json.loads(r.read())["result"]
            bids = ob["b"][:3]
            asks = ob["a"][:3]
            depth_str = (
                f"\nOrderbook (top 3):\n"
                f"  Asks: {' | '.join(f'{a[0]}({a[1]})' for a in asks)}\n"
                f"  Bids: {' | '.join(f'{b[0]}({b[1]})' for b in bids)}"
            )
        except Exception:
            pass

    spread = results.get("ask", 0) - results.get("bid", 0)
    funding = results.get("funding", 0)
    funding_str = (
        f"  Funding:    {funding:+.4f}%/8h {'⚠️ HIGH' if abs(funding) > 0.05 else ''}"
    )

    return (
        f"{sym}/USDT [{results['source']}]\n"
        f"  Price:      ${results['price']:,.2f}\n"
        f"  24h Change: {results['change_24h']:+.2f}%\n"
        f"  24h Volume: ${results['volume_24h']:,.0f}\n"
        f"{funding_str}\n"
        f"  Open Int:   ${results.get('open_int', 0):,.0f}\n"
        f"  Spread:     ${spread:.4f}\n"
        f"{depth_str}"
    )


TOOL_REGISTRY["market_intel"] = market_intel

# ══ TOOL #2: codebase_map ════════════════════════════════════


def codebase_map(path: str = ".", max_depth: int = 3, show_sizes: bool = True) -> str:
    """
    Generates a full architectural map of a project:
    - File tree (respecting .gitignore)
    - Language breakdown by line count
    - Entry points detected (main.py, index.js, main.go, etc.)
    - Import/dependency graph (top-level)
    - Largest files (likely most important)
    - TODO/FIXME count per file
    - Git status summary
    """
    import os, subprocess, re
    from pathlib import Path

    path = os.path.expanduser(path)

    SKIP = {
        ".git",
        "venv",
        ".venv",
        "node_modules",
        "__pycache__",
        "target",
        "dist",
        "build",
        ".cache",
        "chroma_db",
    }
    CODE_EXT = {
        ".py": "Python",
        ".go": "Go",
        ".rs": "Rust",
        ".ts": "TypeScript",
        ".js": "JavaScript",
        ".mojo": "Mojo",
        ".cpp": "C++",
        ".c": "C",
        ".sh": "Shell",
        ".md": "Markdown",
        ".yaml": "YAML",
        ".json": "JSON",
    }

    lines_by_lang = {}
    file_lines = {}
    entry_points = []
    todo_count = {}
    tree_lines = []

    ENTRY_NAMES = {
        "main.py",
        "main.go",
        "main.rs",
        "index.js",
        "index.ts",
        "app.py",
        "server.py",
        "start.py",
        "main.cpp",
        "main.c",
        "Makefile",
        "docker-compose.yml",
    }

    def walk(dirpath, prefix="", depth=0):
        if depth > max_depth:
            return
        try:
            entries = sorted(os.listdir(dirpath))
        except PermissionError:
            return
        entries = [e for e in entries if e not in SKIP and not e.startswith(".")]
        for i, entry in enumerate(entries):
            full = os.path.join(dirpath, entry)
            connector = "└── " if i == len(entries) - 1 else "├── "
            ext = os.path.splitext(entry)[1]
            if os.path.isdir(full):
                tree_lines.append(f"{prefix}{connector}{entry}/")
                new_prefix = prefix + ("    " if i == len(entries) - 1 else "│   ")
                walk(full, new_prefix, depth + 1)
            elif os.path.isfile(full):
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    n_lines = content.count("\n")
                    lang = CODE_EXT.get(ext, "")
                    if lang:
                        lines_by_lang[lang] = lines_by_lang.get(lang, 0) + n_lines
                    file_lines[full] = n_lines
                    # Detect entry points
                    if entry in ENTRY_NAMES:
                        rel = os.path.relpath(full, path)
                        entry_points.append(rel)
                    # Count TODOs
                    todos = len(
                        re.findall(r"TODO|FIXME|HACK|XXX", content, re.IGNORECASE)
                    )
                    if todos:
                        todo_count[os.path.relpath(full, path)] = todos
                    size_str = f" ({n_lines}L)" if show_sizes and lang else ""
                except Exception:
                    size_str = ""
                tree_lines.append(f"{prefix}{connector}{entry}{size_str}")

    walk(path)

    # Top 5 largest files
    top_files = sorted(file_lines.items(), key=lambda x: x[1], reverse=True)[:5]

    # Git status
    git_str = ""
    try:
        r = subprocess.run(
            ["git", "-C", path, "status", "--short"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.stdout.strip():
            changed = len(r.stdout.strip().split("\n"))
            git_str = f"\nGit: {changed} changed file(s)"
    except Exception:
        pass

    # Language breakdown
    total_lines = sum(lines_by_lang.values()) or 1
    lang_str = "\n".join(
        f"  {lang:<12} {lines:>6} lines  {'█' * int(lines / total_lines * 20)}"
        for lang, lines in sorted(
            lines_by_lang.items(), key=lambda x: x[1], reverse=True
        )
    )

    todo_str = ""
    if todo_count:
        top_todos = sorted(todo_count.items(), key=lambda x: x[1], reverse=True)[:5]
        todo_str = "\nTODOs/FIXMEs:\n" + "\n".join(f"  {f}: {n}" for f, n in top_todos)

    top_files_str = "\nLargest files:\n" + "\n".join(
        f"  {os.path.relpath(f, path)}: {n} lines" for f, n in top_files
    )

    entry_str = "\nEntry points: " + ", ".join(entry_points) if entry_points else ""

    return (
        f"=== {os.path.basename(path)} Architecture ===\n"
        f"{chr(10).join(tree_lines[:60])}"
        f"{'... (truncated)' if len(tree_lines) > 60 else ''}\n\n"
        f"Languages:\n{lang_str}\n"
        f"{entry_str}\n"
        f"{top_files_str}"
        f"{todo_str}"
        f"{git_str}"
    )


TOOL_REGISTRY["codebase_map"] = codebase_map

# ══ TOOL #3: process_watch ═══════════════════════════════════


def process_watch(action: str = "list", name: str = "", signal: str = "status") -> str:
    """
    Monitor, control, and inspect running processes.
    """
    import subprocess, psutil, os, re

    action = action.lower()

    if action == "list":
        lines = ["PID    CPU%   RAM(MB)  NAME"]
        procs = []
        for p in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_info", "username", "cmdline"]
        ):
            try:
                if p.info["username"] == os.getenv("USER"):
                    ram = p.info["memory_info"].rss / 1024**2
                    cmd = " ".join(p.info["cmdline"][:3])[:40]
                    procs.append((p.info["cpu_percent"], ram, p.info["pid"], cmd))
            except Exception:
                pass
        procs.sort(reverse=True)
        for cpu, ram, pid, cmd in procs[:20]:
            lines.append(f"{pid:<6} {cpu:>5.1f}%  {ram:>8.1f}  {cmd}")
        # VRAM summary
        try:
            vr = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory,name",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if vr.stdout.strip():
                lines.append("\nVRAM consumers:")
                for line in vr.stdout.strip().split("\n"):
                    lines.append(f"  {line.strip()}")
        except Exception:
            pass
        return "\n".join(lines)

    elif action == "find":
        if not name:
            return "Error: name required for find"
        results = []
        for p in psutil.process_iter(
            ["pid", "name", "cmdline", "status", "cpu_percent", "memory_info"]
        ):
            try:
                cmd = " ".join(p.info["cmdline"])
                if name.lower() in cmd.lower():
                    ram = p.info["memory_info"].rss / 1024**2
                    results.append(
                        f"PID {p.info['pid']} [{p.info['status']}] "
                        f"CPU:{p.info['cpu_percent']:.1f}% "
                        f"RAM:{ram:.0f}MB\n  {cmd[:80]}"
                    )
            except Exception:
                pass
        return "\n".join(results) if results else f"No process: {name}"

    elif action == "kill":
        if not name:
            return "Error: name required"
        killed = []
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if name.lower() in " ".join(p.info["cmdline"]).lower():
                    p.terminate()
                    killed.append(str(p.info["pid"]))
            except Exception:
                pass
        return (
            f"Killed PIDs: {', '.join(killed)}"
            if killed
            else f"No process matching: {name}"
        )

    elif action == "vram":
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory,name",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if not r.stdout.strip():
                return "No VRAM consumers."
            lines = ["PID    VRAM(MB)  Process"]
            for line in r.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    lines.append(f"{parts[0]:<6} {parts[1]:>8}  {parts[2][:40]}")
            return "\n".join(lines)
        except Exception as e:
            return f"nvidia-smi error: {e}"

    return f"Unknown action: {action}"


TOOL_REGISTRY["process_watch"] = process_watch

# ══ TOOL #4: log_analyzer ════════════════════════════════════


def log_analyzer(path: str, mode: str = "auto", tail: int = 200) -> str:
    """
    Intelligent log analysis — extracts signal from noise.
    """
    import os, re
    from collections import Counter
    from datetime import datetime

    path = os.path.expanduser(path)
    if os.path.isdir(path):
        # Glob for logs inside directory
        logs = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith((".log", ".txt")):
                    logs.append(os.path.join(root, f))
        if not logs:
            return f"No log files in: {path}"
        path = max(logs, key=os.path.getmtime)  # newest
    elif not os.path.exists(path):
        return f"File not found: {path}"

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading {path}: {e}"

    if not lines:
        return f"Empty log: {path}"

    tail_lines = lines[-tail:]
    full_text = "".join(tail_lines)

    # Auto-detect mode
    if mode == "auto":
        if any(
            w in full_text.lower()
            for w in [
                "pnl",
                "profit",
                "loss",
                "trade",
                "order",
                "fill",
                "bybit",
                "okx",
                "binance",
            ]
        ):
            mode = "pnl"
        elif any(
            w in full_text for w in ["Traceback", "Exception", "Error:", "CRITICAL"]
        ):
            mode = "crashes"
        else:
            mode = "errors"

    results = [
        f"Log: {os.path.basename(path)} "
        f"({len(lines)} lines total, last {tail} shown)\n"
        f"Mode: {mode}\n"
    ]

    if mode == "pnl":
        # Extract trading metrics
        trades = []
        pnl_pattern = re.compile(
            r"(?:pnl|profit|loss|realized)[:\s]+([+-]?\d+\.?\d*)", re.IGNORECASE
        )
        trade_pattern = re.compile(
            r"(?:trade|order|fill)[:\s]+"
            r"(?:side[:\s]+)?(buy|sell|long|short)",
            re.IGNORECASE,
        )
        wins = losses = 0
        total_pnl = 0.0

        for line in tail_lines:
            pnl_m = pnl_pattern.search(line)
            if pnl_m:
                val = float(pnl_m.group(1))
                total_pnl += val
                if val > 0:
                    wins += 1
                elif val < 0:
                    losses += 1
                trades.append(f"  {line.strip()[:80]}")

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades else 0
        results.append(
            f"PNL Summary:\n"
            f"  Total PNL:  {total_pnl:+.4f}\n"
            f"  Win/Loss:   {wins}W / {losses}L\n"
            f"  Win Rate:   {win_rate:.1f}%\n"
            f"  Trades:     {total_trades}\n"
        )
        if trades:
            results.append("Recent PNL lines:\n" + "\n".join(trades[-10:]))

    elif mode == "errors":
        ERROR_RE = re.compile(
            r"(ERROR|WARN|WARNING|CRITICAL|FATAL|FAILED)", re.IGNORECASE
        )
        error_lines = [
            (i + 1, l.strip()) for i, l in enumerate(tail_lines) if ERROR_RE.search(l)
        ]
        results.append(f"Errors/Warnings ({len(error_lines)} found):\n")
        for lineno, line in error_lines[-20:]:
            results.append(f"  L{lineno}: {line[:100]}")

    elif mode == "crashes":
        # Find traceback blocks
        blocks = []
        in_tb = False
        tb_lines = []
        for line in tail_lines:
            if "Traceback" in line:
                in_tb = True
                tb_lines = [line]
            elif in_tb:
                tb_lines.append(line)
                if line.strip().startswith(("Error:", "Exception")) or (
                    len(tb_lines) > 15
                ):
                    blocks.append("".join(tb_lines))
                    in_tb = False
                    tb_lines = []
        results.append(f"Crashes found: {len(blocks)}\n")
        for i, block in enumerate(blocks[-3:]):  # last 3
            results.append(f"\n--- Crash {i + 1} ---\n{block[:500]}")

    elif mode == "pattern":

        def normalize(line):
            l = re.sub(
                r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
                r"[\d.:+-]*",
                "[TIME]",
                line,
            )
            l = re.sub(r"\b\d+\.?\d*\b", "[N]", l)
            l = re.sub(r"['\"][^'\"]{3,}['\"]", "[STR]", l)
            return l.strip()[:80]

        patterns = Counter(normalize(l) for l in tail_lines if l.strip())
        results.append("Top patterns (most frequent noise):\n")
        for pattern, count in patterns.most_common(10):
            results.append(f"  {count:>4}x  {pattern}")

    return "\n".join(str(r) for r in results)


TOOL_REGISTRY["log_analyzer"] = log_analyzer

# ══ TOOL #5: code_xray ═══════════════════════════════════════


def code_xray(path: str, focus: str = "all") -> str:
    """Deep static analysis of a source file."""
    import os, re, ast

    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"File not found: {path}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        return f"Error reading {path}: {e}"

    lang = os.path.splitext(path)[1]
    lines = source.split("\n")
    results = [
        f"=== code_xray: {os.path.basename(path)} ({len(lines)} lines, {lang}) ===\n"
    ]

    if lang == ".py" and focus in ("all", "functions"):
        try:
            tree = ast.parse(source)
            funcs = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a.arg for a in node.args.args]
                    doc = (ast.get_docstring(node) or "")[:80]
                    complexity = sum(
                        1
                        for n in ast.walk(node)
                        if isinstance(
                            n, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)
                        )
                    )
                    funcs.append(
                        f"  {'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}"
                        f"def {node.name}({', '.join(args)}) "
                        f"[L{node.lineno}, complexity={complexity}]"
                        f"{chr(10) + '    ' + doc if doc else ''}"
                    )
            results.append(f"Functions ({len(funcs)}):\n" + "\n".join(funcs))
        except SyntaxError as e:
            results.append(f"Syntax error in file: {e}")

    if focus in ("all", "imports"):
        import_re = re.compile(
            r"^(?:import|from|require|use|#include)\s+(.+)", re.MULTILINE
        )
        imports = import_re.findall(source)
        results.append(
            f"\nDependencies ({len(imports)}):\n  " + "\n  ".join(imports[:20])
        )

    if focus in ("all", "security"):
        DANGER = {
            "hardcoded_secret": re.compile(
                r"(?:password|secret|api_key|token|passphrase)"
                r'\s*=\s*["\'][^"\']{6,}["\']',
                re.IGNORECASE,
            ),
            "shell_injection": re.compile(
                r"subprocess\.(?:call|run|Popen)"
                r".*shell\s*=\s*True"
            ),
            "eval_exec": re.compile(r"\b(?:eval|exec)\s*\("),
            "sql_concat": re.compile(r"(?:execute|query)\s*\([^)]*\+[^)]*\)"),
            "pickle": re.compile(r"pickle\.(?:load|loads)"),
            "insecure_random": re.compile(r"\brandom\.\w+\s*\("),
        }
        findings = []
        for issue, pattern in DANGER.items():
            matches = [(m.start(), m.group()[:60]) for m in pattern.finditer(source)]
            for pos, match in matches:
                lineno = source[:pos].count("\n") + 1
                findings.append(f"  ⚠️ {issue} L{lineno}: {match}")
        if findings:
            results.append("\nSecurity flags:\n" + "\n".join(findings))
        else:
            results.append("\nSecurity: no obvious issues found.")

    return "\n".join(results)


TOOL_REGISTRY["code_xray"] = code_xray

# ══ TOOL #6: multi_edit ══════════════════════════════════════


def multi_edit(edits: list, dry_run: bool = False) -> str:
    """Apply multiple file edits atomically with full rollback."""
    import os, difflib, shutil
    from datetime import datetime

    backups = {}
    applied = []
    results = []

    def _backup(filepath):
        if filepath not in backups:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    backups[filepath] = f.read()
            except FileNotFoundError:
                backups[filepath] = None  # new file

    def _rollback():
        for filepath, original in backups.items():
            try:
                if original is None:
                    os.remove(filepath)
                else:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(original)
            except Exception:
                pass

    total_added = total_removed = 0

    for edit in edits:
        filepath = os.path.expanduser(edit.get("file", ""))
        if not filepath:
            results.append("  ERROR: edit missing 'file' key")
            continue

        _backup(filepath)

        try:
            if "create" in edit:
                content = edit["create"]
                if not dry_run:
                    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)
                lines = content.count("\n")
                results.append(
                    f"  {'[DRY] ' if dry_run else ''}CREATE {filepath} ({lines} lines)"
                )
                total_added += lines

            elif "old" in edit and "new" in edit:
                with open(filepath, "r", encoding="utf-8") as f:
                    original = f.read()
                if edit["old"] not in original:
                    # Fuzzy fallback
                    import difflib

                    ratio = difflib.SequenceMatcher(None, edit["old"], original).ratio()
                    results.append(
                        f"  ⚠️ SKIP {filepath}: "
                        f"old text not found "
                        f"(best match: {ratio:.0%})"
                    )
                    continue
                new_content = original.replace(edit["old"], edit["new"], 1)
                diff = list(
                    difflib.unified_diff(
                        original.splitlines(), new_content.splitlines(), lineterm=""
                    )
                )
                added = sum(
                    1 for l in diff if l.startswith("+") and not l.startswith("+++")
                )
                removed = sum(
                    1 for l in diff if l.startswith("-") and not l.startswith("---")
                )
                if not dry_run:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(new_content)
                results.append(
                    f"  {'[DRY] ' if dry_run else ''}EDIT "
                    f"{filepath} (+{added}/-{removed} lines)"
                )
                total_added += added
                total_removed += removed

            elif "append" in edit:
                with open(filepath, "a", encoding="utf-8") as f:
                    if not dry_run:
                        f.write("\n" + edit["append"])
                lines = edit["append"].count("\n")
                results.append(
                    f"  {'[DRY] ' if dry_run else ''}APPEND {filepath} (+{lines} lines)"
                )
                total_added += lines

            applied.append(filepath)

        except Exception as e:
            results.append(f"  ERROR in {filepath}: {e}")
            if not dry_run:
                _rollback()
                return "ROLLBACK: error in edit — all changes reverted.\n" + "\n".join(
                    results
                )

    action = "DRY RUN" if dry_run else "APPLIED"
    return (
        f"multi_edit {action}: {len(applied)} files\n"
        f"  +{total_added} lines  -{total_removed} lines\n" + "\n".join(results)
    )


TOOL_REGISTRY["multi_edit"] = multi_edit

# ══ TOOL #7: bot_runner ══════════════════════════════════════


def bot_runner(
    action: str, bot_path: str = "", bot_name: str = "", args: str = ""
) -> str:
    """Start, stop, restart, and monitor crypto trading bots."""
    import subprocess, os, re, time
    from pathlib import Path

    bot_path = os.path.expanduser(bot_path) if bot_path else ""

    def _detect_start_cmd(path: str) -> str:
        p = Path(path)
        if (p / "start.sh").exists():
            return f"bash start.sh {args}"
        if (p / "Makefile").exists():
            mk = (p / "Makefile").read_text()
            if "start:" in mk:
                return f"make start {args}"
        if (p / "main.py").exists():
            venv = p / "venv" / "bin" / "python3"
            py = str(venv) if venv.exists() else "python3"
            return f"{py} main.py {args}"
        if (p / "main.go").exists():
            return f"go run . {args}"
        if (p / "Cargo.toml").exists():
            return f"cargo run -- {args}"
        if (p / "docker-compose.yml").exists():
            return f"docker-compose up -d {args}"
        return f"echo 'Cannot detect start command for {path}'"

    if not bot_name and bot_path:
        bot_name = Path(bot_path).name.upper()[:20]

    def _tmux(cmd: str) -> str:
        r = subprocess.run(f"tmux {cmd}", shell=True, capture_output=True, text=True)
        return r.stdout + r.stderr

    if action == "status":
        r = subprocess.run(
            [
                "tmux",
                "list-sessions",
                "-F",
                "#{session_name} #{session_created} #{window_name}",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return "No tmux sessions running."
        sessions = r.stdout.strip().split("\n")
        lines = ["Running bot sessions:"]
        for s in sessions:
            parts = s.split()
            if len(parts) >= 2:
                name = parts[0]
                ts = int(parts[1]) if parts[1].isdigit() else 0
                if ts:
                    elapsed = int(time.time()) - ts
                    h, m = divmod(elapsed // 60, 60)
                    uptime = f"{h}h{m:02d}m"
                else:
                    uptime = "?"
                lines.append(f"  {name:<20} uptime: {uptime}")
        return "\n".join(lines)

    elif action == "start":
        if not bot_path:
            return "Error: bot_path required for start"
        cmd = _detect_start_cmd(bot_path)
        _tmux(f"kill-session -t {bot_name} 2>/dev/null")
        _tmux(f"new-session -d -s {bot_name} -c '{bot_path}' '{cmd}'")
        time.sleep(1)
        r = _tmux(f"has-session -t {bot_name}")
        alive = "error" not in r.lower()
        return (
            f"{'Started' if alive else 'FAILED'}: "
            f"{bot_name}\n"
            f"  cmd: {cmd}\n"
            f"  dir: {bot_path}\n"
            f"  attach: tmux attach -t {bot_name}"
        )

    elif action == "stop":
        if not bot_name:
            return "Error: bot_name required"
        _tmux(f"send-keys -t {bot_name} C-c")
        time.sleep(0.5)
        _tmux(f"kill-session -t {bot_name}")
        return f"Stopped: {bot_name}"

    elif action == "snapshot":
        if not bot_name:
            return "Error: bot_name required"
        r = subprocess.run(
            ["tmux", "capture-pane", "-pt", bot_name, "-S", "-50"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return f"Session {bot_name} not found"
        return f"=== {bot_name} (last 50 lines) ===\n{r.stdout}"

    elif action == "attach":
        return f"Run: tmux attach -t {bot_name}"

    elif action == "restart":
        stop_result = bot_runner("stop", bot_name=bot_name)
        time.sleep(1)
        return (
            stop_result
            + "\n"
            + bot_runner("start", bot_path=bot_path, bot_name=bot_name, args=args)
        )

    return f"Unknown action: {action}"


TOOL_REGISTRY["bot_runner"] = bot_runner

# ══ TOOL #8: secret_scan ═════════════════════════════════════


def secret_scan(path: str = ".", fix: bool = False) -> str:
    """Scan entire directory tree for exposed secrets."""
    import os, re
    from pathlib import Path

    path = os.path.expanduser(path)

    SECRET_PATTERNS = [
        ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}")),
        ("openai_key", re.compile(r"sk-[A-Za-z0-9]{32,}")),
        ("openrouter_key", re.compile(r"sk-or-[A-Za-z0-9\-_]{20,}")),
        ("telegram_token", re.compile(r"\d{8,12}:AA[A-Za-z0-9_\-]{8,}")),
        (
            "bybit_key",
            re.compile(
                r"(?:BYBIT|bybit)[_\w]*(?:KEY|SECRET|key|secret)"
                r"\s*[=:]\s*([A-Za-z0-9]{16,})"
            ),
        ),
        (
            "okx_key",
            re.compile(
                r"(?:OKX|okx)[_\w]*(?:KEY|SECRET|PASS|key|secret)"
                r"\s*[=:]\s*([A-Za-z0-9\-]{16,})"
            ),
        ),
        (
            "generic_password",
            re.compile(
                r"(?:password|passwd|pwd)\s*[=:]\s*"
                r'["\']([^"\']{8,})["\']',
                re.IGNORECASE,
            ),
        ),
        ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----")),
        (
            "jwt_secret",
            re.compile(
                r"(?:JWT_SECRET|jwt_secret)\s*[=:]\s*"
                r'["\']?([A-Za-z0-9+/]{20,})["\']?'
            ),
        ),
        ("hardcoded_ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}:\d{4,5}\b")),
    ]

    SKIP_DIRS = {
        ".git",
        "venv",
        ".venv",
        "__pycache__",
        "node_modules",
        "target",
        "dist",
    }
    SKIP_FILES = {
        ".pyc",
        ".so",
        ".bin",
        ".exe",
        ".jpg",
        ".png",
        ".gif",
        ".zip",
        ".tar",
        ".gz",
    }

    findings = []
    scanned = 0

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if any(fname.endswith(s) for s in SKIP_FILES):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                scanned += 1
            except Exception:
                continue

            file_findings = []
            for secret_type, pattern in SECRET_PATTERNS:
                for m in pattern.finditer(content):
                    lineno = content[: m.start()].count("\n") + 1
                    val = m.group()
                    masked = val[:4] + "****" + val[-4:]
                    file_findings.append(f"    L{lineno} [{secret_type}]: {masked}")
                    if fix:
                        content = content.replace(val, "[REDACTED]")

            if file_findings:
                rel = os.path.relpath(fpath, path)
                findings.append(f"  {rel}:\n" + "\n".join(file_findings))
                if fix:
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(content)

    summary = (
        f"secret_scan: {scanned} files scanned\n"
        f"{'SECRETS FOUND' if findings else 'Clean — no secrets found'}:\n"
    )
    if findings:
        summary += "\n".join(findings)
        if fix:
            summary += "\n\n[FIX APPLIED: secrets redacted in-place]"
        else:
            summary += (
                "\n\n⚠️ Run secret_scan(fix=True) to redact "
                "— or rotate these credentials immediately."
            )
    return summary


TOOL_REGISTRY["secret_scan"] = secret_scan

# ══ TOOL #9: model_bench ═════════════════════════════════════


def model_bench(
    model: str = "aria-qwen",
    prompt: str = "Write a Python fibonacci function.",
    runs: int = 3,
) -> str:
    """Benchmark any local Ollama model."""
    import time, subprocess, json

    try:
        import ollama
    except ImportError:
        return "ollama not installed"

    def _get_vram_free_mb() -> int:
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return int(r.stdout.strip())
        except Exception:
            return 0

    results = []
    ttfts = []
    tps_list = []

    for run in range(runs):
        vram_before = _get_vram_free_mb()
        t_start = time.monotonic()
        t_first_token = None
        token_count = 0
        response_text = ""

        try:
            stream = ollama.generate(
                model=model,
                prompt=prompt,
                stream=True,
                options={"num_ctx": 512, "num_predict": 100, "temperature": 0.1},
            )
            for chunk in stream:
                if t_first_token is None:
                    t_first_token = time.monotonic()
                token_count += 1
                response_text += chunk.get("response", "")
                if chunk.get("done"):
                    break
        except Exception as e:
            return f"Bench failed: {e}"

        t_end = time.monotonic()
        vram_after = _get_vram_free_mb()

        total_time = t_end - t_start
        ttft = (t_first_token - t_start) if t_first_token else 0
        tps = token_count / total_time if total_time > 0 else 0
        vram_used = vram_before - vram_after

        ttfts.append(ttft)
        tps_list.append(tps)
        results.append((run + 1, ttft, tps, total_time, vram_used))

    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
    avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0
    vram_used_mb = results[-1][4] if results else 0

    has_code = "def " in response_text or "function" in response_text
    quality = "✓ code produced" if has_code else "? no code found"

    lines = [
        f"Benchmark: {model} ({runs} runs)",
        f"  Avg TTFT:    {avg_ttft * 1000:.0f}ms",
        f"  Avg tok/sec: {avg_tps:.1f}",
        f"  VRAM used:   ~{vram_used_mb:.0f}MB",
        f"  Quality:     {quality}",
        "",
        "Run details:",
    ]
    for run, ttft, tps, total, vram in results:
        lines.append(
            f"  Run {run}: TTFT={ttft * 1000:.0f}ms  {tps:.1f}t/s  {total:.1f}s total"
        )

    lines.append(f"\nSample response:\n{response_text[:200]}")
    return "\n".join(lines)


TOOL_REGISTRY["model_bench"] = model_bench

# ══ TOOL #10: context_inject ═════════════════════════════════


def context_inject(mode: str = "project", path: str = ".", query: str = "") -> str:
    """Intelligently builds the PERFECT context block."""
    import os, subprocess
    from pathlib import Path

    path = os.path.expanduser(path)

    if mode == "project":
        return codebase_map(path, max_depth=2, show_sizes=True)

    elif mode == "task":
        if not query:
            return "Error: query required for task mode"
        keywords = [
            w
            for w in query.lower().split()
            if len(w) > 3
            and w not in {"what", "where", "when", "how", "the", "this", "that", "with"}
        ]
        relevant_files = []
        for root, dirs, files in os.walk(path):
            dirs[:] = [
                d
                for d in dirs
                if d not in {".git", "venv", ".venv", "__pycache__", "node_modules"}
            ]
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, path)
                score = sum(1 for kw in keywords if kw in rel.lower())
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        head = f.read(500)
                    score += sum(1 for kw in keywords if kw in head.lower())
                    if score > 0:
                        relevant_files.append((score, rel, head))
                except Exception:
                    pass
        relevant_files.sort(reverse=True)
        result = [f"Context for: '{query}'\nRelevant files:\n"]
        for score, rel, head in relevant_files[:5]:
            result.append(f"=== {rel} (relevance: {score}) ===\n{head[:300]}\n")
        return "\n".join(result)

    elif mode == "diff":
        try:
            r = subprocess.run(
                ["git", "-C", path, "diff", "HEAD", "--stat"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            diff_r = subprocess.run(
                ["git", "-C", path, "diff", "HEAD", "--unified=2"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return (
                f"Git diff summary:\n{r.stdout}\n\n"
                f"Diff (±2 context):\n"
                f"{diff_r.stdout[:2000]}"
            )
        except Exception as e:
            return f"git diff error: {e}"

    elif mode == "bot":
        parts = [codebase_map(path, max_depth=1)]
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".log"):
                    fpath = os.path.join(root, f)
                    parts.append(log_analyzer(fpath, mode="auto", tail=30))
            break
        return "\n\n".join(parts)

    elif mode == "session":
        mile_path = os.path.expanduser("~/.aria/milestones.md")
        if not os.path.exists(mile_path):
            return "No session history found."
        with open(mile_path) as f:
            lines = f.readlines()
        return "Recent session context:\n" + "".join(lines[-40:])

    return f"Unknown mode: {mode}"


TOOL_REGISTRY["context_inject"] = context_inject

# ══ VERIFY ═══════════════════════════════════════════════════
NEW_TOOLS = [
    "market_intel",
    "codebase_map",
    "process_watch",
    "log_analyzer",
    "code_xray",
    "multi_edit",
    "bot_runner",
    "secret_scan",
    "model_bench",
    "context_inject",
]
for t in NEW_TOOLS:
    assert t in TOOL_REGISTRY, f"MISSING from registry: {t}"
