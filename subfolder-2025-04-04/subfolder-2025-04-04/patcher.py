import re

# Patch tools.py
with open("tools.py", "r") as f:
    tools_content = f.read()

sanitizer_code = r"""
import re as _re

# Patterns that must NEVER appear in output
_SECRET_PATTERNS = [
    # API keys (various formats)
    (_re.compile(r'((?:API_KEY|SECRET|TOKEN|PASSWORD|PASSPHRASE|PRIVATE_KEY)'
                 r'\s*[=:]\s*)([^\s\n]{8,})', _re.IGNORECASE),
     lambda m: m.group(1) + '[REDACTED]'),

    # Telegram bot tokens
    (_re.compile(r'\b(\d{8,12}:AA[A-Za-z0-9_\-]{30,})\b'),
     lambda m: '[TELEGRAM_TOKEN_REDACTED]'),

    # Anthropic keys
    (_re.compile(r'\bsk-ant-[A-Za-z0-9\-_]{20,}\b'),
     lambda m: '[ANTHROPIC_KEY_REDACTED]'),

    # OpenRouter keys
    (_re.compile(r'\bsk-or-[A-Za-z0-9\-_]{20,}\b'),
     lambda m: '[OPENROUTER_KEY_REDACTED]'),

    # Bybit/OKX/exchange keys (alphanumeric 16-40 chars after KEY= or SECRET=)
    (_re.compile(r'((?:BYBIT|OKX|BINANCE|KRAKEN)[_\w]*(?:KEY|SECRET|PASS)'
                 r'\s*=\s*)([A-Za-z0-9]{10,})', _re.IGNORECASE),
     lambda m: m.group(1) + '[EXCHANGE_KEY_REDACTED]'),

    # Tavily keys
    (_re.compile(r'\btvly-[A-Za-z0-9\-_]{10,}\b'),
     lambda m: '[TAVILY_KEY_REDACTED]'),

    # Phone numbers
    (_re.compile(r'\+\d{7,15}\b'),
     lambda m: '+[PHONE_REDACTED]'),
]

def sanitize_output(text: str) -> str:
    \"\"\"Mask all secrets from any output before displaying to user.\"\"\"
    if not text:
        return text
    for pattern, replacer in _SECRET_PATTERNS:
        text = pattern.sub(replacer, text)
    return text

class SessionState:
    \"\"\"Tracks mutable session state across tool calls.\"\"\"
    def __init__(self):
        import os
        self.cwd = os.getcwd()
        self.read_files = {}    # path -> content (cache)
        self.env_vars = {}      # key -> masked value
        self.created_files = [] # files created this session
        self.models_in_vram = []  # models currently loaded

    def update_cwd(self, new_cwd: str):
        import os
        expanded = os.path.expanduser(new_cwd)
        if os.path.isdir(expanded):
            self.cwd = expanded
        return self.cwd

    def record_file_read(self, path: str, content: str):
        self.read_files[path] = content[:200]  # store preview

    def was_read(self, path: str) -> bool:
        return path in self.read_files

SESSION = SessionState()

def read_env_safe(path: str) -> str:
    \"\"\"
    Read .env file but mask all values, show only key names.
    Use this instead of read_file() for .env files.
    \"\"\"
    import os
    try:
        with open(os.path.expanduser(path)) as f:
            lines = f.readlines()
        masked = []
        for line in lines:
            line = line.rstrip()
            if not line or line.startswith('#'):
                masked.append(line)
                continue
            if '=' in line:
                key, _, val = line.partition('=')
                if val and not val.startswith('#'):
                    preview = val[:3] + '...' if len(val) > 3 else '***'
                    masked.append(f"{key}=[MASKED — {len(val)} chars, preview: {preview}]")
                else:
                    masked.append(line)
            else:
                masked.append(line)
        return '\n'.join(masked)
    except Exception as e:
        return f"Error reading {path}: {e}"

def check_model_available(model_path: str) -> dict:
    \"\"\"
    Check if a model is available locally before using it.
    Returns {"available": bool, "location": str, "vram_mb": int}
    \"\"\"
    import os, subprocess

    # Check Ollama models
    try:
        r = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if model_path.lower() in r.stdout.lower():
            return {"available": True, "location": "ollama", "vram_mb": 0}
    except Exception:
        pass

    # Check HuggingFace cache
    hf_cache = os.path.expanduser('~/.cache/huggingface/hub')
    model_dir_name = 'models--' + model_path.replace('/', '--')
    hf_path = os.path.join(hf_cache, model_dir_name)
    if os.path.exists(hf_path):
        # Estimate size
        try:
            r = subprocess.run(['du', '-sm', hf_path],
                             capture_output=True, text=True)
            size_mb = int(r.stdout.split()[0]) if r.stdout else 0
        except Exception:
            size_mb = 0
        return {"available": True, "location": hf_path, "vram_mb": size_mb}

    return {"available": False, "location": None, "vram_mb": 0}

def get_available_models() -> str:
    \"\"\"List all models available locally (Ollama + HuggingFace cache).\"\"\"
    import subprocess, os

    lines = ["=== Available Models ==="]

    # Ollama models
    try:
        r = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if r.stdout.strip():
            lines.append("Ollama:")
            for line in r.stdout.strip().split('\n')[1:]:  # skip header
                lines.append(f"  {line.split()[0]}")
    except Exception:
        lines.append("Ollama: (not running)")

    # HuggingFace cache
    hf_cache = os.path.expanduser('~/.cache/huggingface/hub')
    if os.path.exists(hf_cache):
        lines.append("HuggingFace cache:")
        for d in os.listdir(hf_cache):
            if d.startswith('models--'):
                model_name = d[8:].replace('--', '/')
                lines.append(f"  {model_name}")

    return '\n'.join(lines)

def get_vram_free_mb() -> int:
    \"\"\"Return free VRAM in MB.\"\"\"
    import subprocess
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True)
        return int(r.stdout.strip())
    except Exception:
        return 0

def check_vram_before_load(model_size_mb: int) -> str:
    \"\"\"
    Before loading any model to GPU, check if there's room.
    Returns "OK" or an error message with alternatives.
    \"\"\"
    free = get_vram_free_mb()
    # Need 20% headroom above model size
    needed = int(model_size_mb * 1.2)
    if free >= needed:
        return f"OK — {free}MB free, {needed}MB needed"
    return (f"VRAM insufficient: {free}MB free, {needed}MB needed. "
            f"Options: use CPU (device='cpu'), unload aria-qwen first "
            f"(ollama stop aria-qwen), or use smaller model.")

def _parse_kv(text: str) -> dict:
    \"\"\"Parse key=value or key: value pairs as fallback.\"\"\"
    import re
    args = {}
    for m in re.finditer(r'(\w+)\s*[=:]\s*["\']?([^"\'<\n]+)["\']?', text):
        args[m.group(1).strip()] = m.group(2).strip()
    return args
"""

if "_SECRET_PATTERNS" not in tools_content:
    tools_content = sanitizer_code + "\n" + tools_content

# Patch parse_tool_calls
old_parse_pattern = re.compile(
    r"def parse_tool_calls\(text: str\) -> list:.*?return results", re.DOTALL
)
new_parse = r"""def parse_tool_calls(text: str) -> list:
    \"\"\"
    Extract tool calls handling ALL format variants Aria might generate:
    - Correct:  <tool name="bash">{"cmd": "ls"}</tool>
    - Broken 1: <tool name="bash"><parameter=cmd>ls</parameter>
    - Broken 2: <tool_call>{"name": "bash", "cmd": "ls"}</tool_call>
    - Broken 3: ```bash\nls\n```  (markdown code blocks)
    \"\"\"
    import re, json
    results = []

    # Format 1: Correct JSON format
    pattern1 = re.compile(
        r'<tool\s+name=["\']([^"\']+)["\']>\s*(.*?)\s*</tool>',
        re.DOTALL | re.IGNORECASE
    )
    for m in pattern1.finditer(text):
        name = m.group(1).strip()
        args_raw = m.group(2).strip()
        try:
            args = json.loads(args_raw)
        except Exception:
            # Try key=value fallback
            args = _parse_kv(args_raw)
        results.append({"name": name, "args": args})

    if results:
        return results

    # Format 2: Broken <parameter=X> format
    pattern2 = re.compile(
        r'<tool\s+name=["\']([^"\']+)["\']>.*?<parameter[= ]+(\w+)>\s*(.*?)\s*(?:</parameter>|$)',
        re.DOTALL | re.IGNORECASE
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
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
        re.DOTALL | re.IGNORECASE
    )
    for m in pattern3.finditer(text):
        try:
            blob = json.loads(m.group(1))
            name = blob.pop('name', blob.pop('tool', 'run_bash'))
            results.append({"name": name, "args": blob})
        except Exception:
            pass

    if results:
        return results

    # Format 4: Bare ```bash``` code blocks (treat as run_bash)
    pattern4 = re.compile(r'```(?:bash|sh|shell)\n(.*?)```', re.DOTALL)
    for m in pattern4.finditer(text):
        cmd = m.group(1).strip()
        if cmd:
            results.append({"name": "run_bash", "args": {"cmd": cmd}})

    return results"""
if "Format 4:" not in tools_content:
    tools_content = old_parse_pattern.sub(new_parse, tools_content)

run_bash_orig = """def run_bash(cmd: str, timeout: int = 15) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout + result.stderr
        if len(output) > 2000:
            output = output[:2000] + "\\n...[output truncated]"
        return output
    except Exception as e:
        return f"Error running bash: {e}\""""

run_bash_new = """def run_bash(cmd: str, timeout: int = 15) -> str:
    try:
        import re as _re
        cd_match = _re.search(r'\\bcd\\s+([^\\s;&|]+)', cmd)
        if cd_match:
            new_dir = cd_match.group(1).strip()
            import os
            resolved = os.path.normpath(
                os.path.join(SESSION.cwd, os.path.expanduser(new_dir)))
            SESSION.update_cwd(resolved)

        full_cmd = f"cd {SESSION.cwd} && {cmd}"
        
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout + result.stderr
        if len(output) > 2000:
            output = output[:2000] + "\\n...[output truncated]"
        return sanitize_output(output)
    except Exception as e:
        return sanitize_output(f"Error running bash: {e}")"""

if "SESSION.cwd" not in tools_content:
    tools_content = tools_content.replace(run_bash_orig, run_bash_new)

read_file_orig = """def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}\""""
read_file_new = """def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            SESSION.record_file_read(path, content)
            return sanitize_output(content)
    except Exception as e:
        return sanitize_output(f"Error reading file: {e}")"""
if "SESSION.record_file_read" not in tools_content:
    tools_content = tools_content.replace(read_file_orig, read_file_new)

tool_registry_addons = """
    "read_env": lambda args: read_env_safe(args.get('path', '.env')),
    "get_cwd": lambda args: f"Current directory: {SESSION.cwd}",
    "check_model": lambda args: str(check_model_available(args.get('model', ''))),
    "list_models": lambda args: get_available_models(),
    "check_vram": lambda args: check_vram_before_load(args.get('model_size_mb', 6000)),"""
if '"read_env"' not in tools_content:
    tools_content = tools_content.replace(
        '"probe_tools": lambda args: str(probe_tools()),',
        '"probe_tools": lambda args: str(probe_tools()),' + tool_registry_addons,
    )

with open("tools.py", "w") as f:
    f.write(tools_content)

# PATCH RAG.PY
with open("rag.py", "r") as f:
    rag_content = f.read()

rag_stream_old_1 = """                # Strip emoji from visible output (backstop for Modelfile rule)
                visible = _EMOJI_RE.sub("", visible)

                if visible:"""
rag_stream_new_1 = """                # Strip emoji from visible output (backstop for Modelfile rule)
                visible = _EMOJI_RE.sub("", visible)
                
                from tools import sanitize_output
                visible = sanitize_output(visible)

                if visible:"""
if "sanitize_output(visible)" not in rag_content:
    rag_content = rag_content.replace(rag_stream_old_1, rag_stream_new_1)

rag_stream_old_2 = """                    # Strip emoji from visible output (backstop for Modelfile rule)
                    visible = _EMOJI_RE.sub("", visible)
    
                    if visible:"""
rag_stream_new_2 = """                    # Strip emoji from visible output (backstop for Modelfile rule)
                    visible = _EMOJI_RE.sub("", visible)
    
                    from tools import sanitize_output
                    visible = sanitize_output(visible)

                    if visible:"""
if rag_stream_old_2 in rag_content:
    rag_content = rag_content.replace(rag_stream_old_2, rag_stream_new_2)

with open("rag.py", "w") as f:
    f.write(rag_content)

# PATCH MODELFILE
with open("Modelfile", "r") as f:
    mf_lines = f.readlines()

new_mf_lines = []
for line in mf_lines:
    if line.startswith("## OWNERSHIP RULE:"):
        break
    new_mf_lines.append(line)

additional_modelfile = """## OWNERSHIP RULE:
You are Aria — an AI assistant. You do NOT own, author, or maintain any of H2's
projects. When H2 asks "is X yours?" the answer is always "No, that's H2's project.
I'm helping build/fix it." Never claim ownership of H2's code, folders, or systems.

## READ-ONCE CACHE RULE:
If you have already read a file in this session and got content from it, that content
is still true. Never say a file "doesn't exist" after successfully reading it.
Before saying any file is missing: check your recent tool results in the conversation.

## ONE THING AT A TIME RULE:
When building a new project or feature:
  Step 1: Create ONE file. Test it works. Print "File X: OK"
  Step 2: Create the NEXT file. Test it. Print "File Y: OK"
  Never create 5+ files in one block. If one fails, all fail silently.

## SHELL COMMAND SIMPLICITY RULE:
For writing files, use Python inline, never heredocs:
  python3 -c "open('file.py', 'w').write('''content''')"
  OR: write the tool call: <tool name="write_file">{"path": "x.py", "content": "..."}</tool>
  NEVER: cat << 'EOF' ... EOF inside && chains — heredocs break in tool calls.

## LOCAL MODEL RULE:
Before using any model path:
  1. <tool name="list_models">{}</tool>
  2. Verify the model appears in results
  3. <tool name="check_vram">{"model_size_mb": ESTIMATED_SIZE}</tool>
  4. Only then load it
  Never invent model paths. Never load to GPU without checking VRAM first.
  aria-qwen uses ~5500MB VRAM. Remaining free: ~2000MB max for second model.
  Any second model must use device="cpu" or load after unloading aria-qwen.

## NO RE-ASKING RULE (REINFORCED):
Check the last 6 messages before asking ANY question.
If the answer exists anywhere in recent context: use it, don't ask.
Specific triggers to NEVER re-ask:
  - Which exchange? (H2 said BYBIT)
  - Which folder? (already established)
  - Demo or live? (H2 said demo)
  - API keys? (in ~/crypto_scalper/.env — use read_env tool)

## PATTERN-COPY ENFORCEMENT:
When building a new connector/integration:
  1. Find existing working implementation: grep -r "bybit\\|bybit_v5" ~/crypto_scalper/ --include="*.py" -l
  2. Read the best example file fully
  3. Extract the exact client init pattern, endpoint URLs, auth method
  4. Replicate — do not invent from scratch
  You have working Bybit V5 code in HYDRA/. Use it. Don't write new from memory.
\"\"\"
"""
# Strip last """ and replace with additional
new_content = "".join(new_mf_lines)
if new_content.endswith('"""\n'):
    new_content = new_content[:-4] + additional_modelfile
elif new_content.endswith('"""'):
    new_content = new_content[:-3] + additional_modelfile
else:
    new_content += additional_modelfile

with open("Modelfile", "w") as f:
    f.write(new_content)
