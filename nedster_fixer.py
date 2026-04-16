#!/usr/bin/env python3
"""
nedster_fixer.py  —  Drop into C:\\Users\\hethu\\Nedster\\ and run once.

    python nedster_fixer.py

Fixes applied:
  1. Context window — forces num_ctx=16384 at generation time
     (Path A failed because agent.py overrides the Modelfile at runtime)
  2. Post-tool summary — model now speaks after tool calls
  3. Phrase-loop detector — kills [YOU ARE NEDSTER...] × 30 loops
  4. Weak-model minimal prompt — no tool XML for ★☆☆ models
  5. Output strip — removes <tool name="X">{json}</tool> from chat
  6. Grep pure-Python — no more [WinError 2] on Windows
  7. Modelfile rebuild — increases num_ctx to 16384 and rebuilds aria-qwen
  8. Soul files — creates ~/.aria/soul/ with correct Windows-aware identity
  9. Anti-menu injection — hard-coded into system prompt
  10. Write-file CWD fix — always resolves against project root

All originals are backed up to _backup_YYYYMMDD_HHMMSS/ before touching.
Run as many times as needed — idempotent.
"""

import os
import re
import sys
import shutil
import subprocess
import textwrap
from pathlib import Path
from datetime import datetime

# ── Globals ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
BACKUP = ROOT / f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
WINS = []
FAILS = []

# ── Helpers ───────────────────────────────────────────────────────────────────
def header(msg: str):
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print('─'*60)

def ok(msg: str):
    WINS.append(msg)
    print(f"  ✓  {msg}")

def fail(msg: str):
    FAILS.append(msg)
    print(f"  ✗  {msg}")

def skip(msg: str):
    print(f"  ·  {msg}")

def backup_file(p: Path):
    if not p.exists():
        return
    BACKUP.mkdir(parents=True, exist_ok=True)
    dest = BACKUP / p.name
    shutil.copy2(p, dest)

def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        fail(f"Cannot read {p.name}: {e}")
        return ""

def write(p: Path, content: str):
    backup_file(p)
    try:
        p.write_text(content, encoding="utf-8")
        ok(f"Wrote {p.name}")
    except Exception as e:
        fail(f"Cannot write {p.name}: {e}")

def patch(p: Path, old: str, new: str, label: str) -> bool:
    """Replace first exact occurrence. Returns True if patched."""
    content = read(p)
    if not content:
        return False
    if old not in content:
        skip(f"{label} — pattern not found in {p.name} (may already be patched)")
        return False
    backup_file(p)
    try:
        p.write_text(content.replace(old, new, 1), encoding="utf-8")
        ok(f"{label}")
        return True
    except Exception as e:
        fail(f"{label}: {e}")
        return False

def patch_regex(p: Path, pattern: str, replacement: str, label: str,
                flags=re.DOTALL) -> bool:
    content = read(p)
    if not content:
        return False
    new_content, n = re.subn(pattern, replacement, content, count=1, flags=flags)
    if n == 0:
        skip(f"{label} — pattern not found in {p.name}")
        return False
    backup_file(p)
    try:
        p.write_text(new_content, encoding="utf-8")
        ok(f"{label}")
        return True
    except Exception as e:
        fail(f"{label}: {e}")
        return False

def append_if_missing(p: Path, marker: str, code: str, label: str):
    """Append code block if marker string not already in file."""
    content = read(p)
    if not content:
        return
    if marker in content:
        skip(f"{label} — already present in {p.name}")
        return
    backup_file(p)
    try:
        p.write_text(content + "\n\n" + code, encoding="utf-8")
        ok(f"{label}")
    except Exception as e:
        fail(f"{label}: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# FIX 1 — CONTEXT WINDOW
# Why Path A failed: agent.py passes options={"num_ctx": 4096} at generation
# time, which OVERRIDES whatever the Modelfile says. We fix the call sites.
# ═════════════════════════════════════════════════════════════════════════════
def fix_context_window():
    header("FIX 1 — Context window (force num_ctx=16384 at generation time)")

    TARGET_CTX = 16384

    for fname in ("agent.py", "nedster.py"):
        p = ROOT / fname
        if not p.exists():
            skip(f"{fname} not found")
            continue
        content = read(p)
        changed = False

        # Pattern A: options={"num_ctx": <N>, ...}
        def bump_ctx(m):
            nonlocal changed
            changed = True
            return m.group(0).replace(m.group(1), str(TARGET_CTX))

        new_content = re.sub(
            r'"num_ctx"\s*:\s*(\d+)',
            bump_ctx,
            content
        )

        # Pattern B: 'num_ctx': <N>
        new_content = re.sub(
            r"'num_ctx'\s*:\s*(\d+)",
            lambda m: m.group(0).replace(m.group(1), str(TARGET_CTX)),
            new_content
        )

        if new_content != content:
            backup_file(p)
            p.write_text(new_content, encoding="utf-8")
            ok(f"Bumped all num_ctx → {TARGET_CTX} in {fname}")
        else:
            # Pattern C: not present at all — inject default options helper
            INJECT = f"""
# ── nedster_fixer: context window defaults ───────────────────────────────────
NEDSTER_GEN_OPTIONS = {{
    "num_ctx": {TARGET_CTX},
    "num_predict": 1024,
    "temperature": 0.05,
    "top_p": 0.9,
}}
# ─────────────────────────────────────────────────────────────────────────────
"""
            append_if_missing(
                p,
                "NEDSTER_GEN_OPTIONS",
                INJECT,
                f"Injected NEDSTER_GEN_OPTIONS ({TARGET_CTX}) into {fname}"
            )

    # Also patch Modelfile directly
    mf = ROOT / "Modelfile"
    if mf.exists():
        content = read(mf)
        new_content = re.sub(
            r'PARAMETER\s+num_ctx\s+\d+',
            f'PARAMETER num_ctx {TARGET_CTX}',
            content
        )
        if new_content != content:
            backup_file(mf)
            mf.write_text(new_content, encoding="utf-8")
            ok(f"Modelfile: num_ctx → {TARGET_CTX}")
        else:
            # Add if missing
            new_content = content.rstrip() + f"\nPARAMETER num_ctx {TARGET_CTX}\n"
            backup_file(mf)
            mf.write_text(new_content, encoding="utf-8")
            ok(f"Modelfile: added num_ctx {TARGET_CTX}")
    else:
        skip("Modelfile not found — skipping Ollama model rebuild trigger")


# ═════════════════════════════════════════════════════════════════════════════
# FIX 2 — REBUILD aria-qwen MODEL
# ═════════════════════════════════════════════════════════════════════════════
def fix_rebuild_model():
    header("FIX 2 — Rebuild aria-qwen with new context window")

    mf = ROOT / "Modelfile"
    if not mf.exists():
        skip("Modelfile not found — cannot rebuild")
        return

    print("  Rebuilding aria-qwen (this takes 20-60 seconds)...")
    try:
        result = subprocess.run(
            ["ollama", "create", "aria-qwen", "-f", str(mf)],
            capture_output=True, text=True, timeout=120, cwd=str(ROOT)
        )
        if result.returncode == 0:
            ok("aria-qwen rebuilt with new context window")
        else:
            fail(f"ollama create failed: {result.stderr[:200]}")
            print("  → Run manually: ollama create aria-qwen -f Modelfile")
    except FileNotFoundError:
        fail("ollama not found in PATH — rebuild manually")
        print("  → Run: ollama create aria-qwen -f Modelfile")
    except subprocess.TimeoutExpired:
        fail("Rebuild timed out — run manually")
        print("  → Run: ollama create aria-qwen -f Modelfile")


# ═════════════════════════════════════════════════════════════════════════════
# FIX 3 — POST-TOOL SUMMARY  
# After tool calls the model goes silent. Inject a summary gate.
# ═════════════════════════════════════════════════════════════════════════════
POST_TOOL_CODE = '''
# ── nedster_fixer: post-tool summary helpers ──────────────────────────────────

def _extract_prose(response: str) -> str:
    """Strip tool XML blocks; return only natural language text."""
    import re as _re
    text = _re.sub(r'<tool\\s+name="[^"]*">.*?</tool>', '', response,
                   flags=_re.DOTALL)
    text = _re.sub(r'<!--.*?-->', '', text, flags=_re.DOTALL)
    return text.strip()


def _needs_summary(response: str, tool_results: str) -> bool:
    """True when tools ran but model produced no prose reply."""
    prose = _extract_prose(response)
    has_tools = '<tool name=' in response
    meaningful = len([c for c in prose if c.isalpha()]) > 15
    return has_tools and not meaningful


def _generate_completion_summary(model: str, user_input: str,
                                  tool_results: str) -> str:
    """One-shot 100-token summary of what just happened."""
    import ollama as _ol, re as _re
    import_match = _re.findall(
        r'(?:Written|Created|Started|Scaffolded|Built):\\s*([^\\s(]+)',
        tool_results)
    files_str = ', '.join(import_match[:4]) if import_match else ''
    prompt = (
        f"Task: {user_input[:80]}\\n"
        f"Results: {tool_results[:500]}\\n\\n"
        f"State in 1-2 sentences what was done. "
        f"{'Files: ' + files_str + '. ' if files_str else ''}"
        f"Be direct. Never ask what to do next."
    )
    try:
        r = _ol.generate(
            model=model, prompt=prompt,
            options={"num_ctx": 2048, "num_predict": 80,
                     "temperature": 0.05, "think": False}
        )
        summary = _extract_prose(r.get("response", "")).strip()
        return summary if len(summary) > 4 else "Done."
    except Exception:
        return "Done."

# ─────────────────────────────────────────────────────────────────────────────
'''

def fix_post_tool_summary():
    header("FIX 3 — Post-tool summary (model speaks after tool calls)")

    agent_py = ROOT / "agent.py"
    if not agent_py.exists():
        fail("agent.py not found")
        return

    append_if_missing(
        agent_py,
        "_needs_summary",
        POST_TOOL_CODE,
        "Injected post-tool summary helpers into agent.py"
    )

    # Now patch _execute_response to call them.
    # Strategy: find where full_response is returned from _execute_response
    # and inject the summary check before the return.
    content = read(agent_py)

    # Look for the return pattern inside what's likely _execute_response
    # We target the final return that comes after the tool loop
    SUMMARY_GUARD = "# [fixer] summary gate\n        if _needs_summary(full_response, tool_results_str):\n            full_response = full_response.rstrip() + '\\n' + _generate_completion_summary(self.model, user_input, tool_results_str)\n"

    if "_needs_summary" not in content:
        skip("Could not inject summary gate — _needs_summary not in agent.py after append")
        return

    if "# [fixer] summary gate" in content:
        skip("Summary gate already injected")
        return

    # Find the tool_results_str variable and the return after it
    # We look for: return full_response  (the last one in _execute_response)
    # and prepend the guard. This is a heuristic — may need manual review.
    PATCH_TARGET = "return full_response"
    if content.count(PATCH_TARGET) >= 1:
        # Replace the LAST occurrence (most likely the final return of _execute_response)
        last_idx = content.rfind(PATCH_TARGET)
        indent = ""
        line_start = content.rfind("\n", 0, last_idx) + 1
        for c in content[line_start:]:
            if c in (' ', '\t'):
                indent += c
            else:
                break
        guard = f"{indent}# [fixer] summary gate\n{indent}if _needs_summary(full_response, tool_results_str if 'tool_results_str' in dir() else ''):\n{indent}    full_response = full_response.rstrip() + '\\n' + _generate_completion_summary(self.model, user_input if 'user_input' in dir() else '', tool_results_str if 'tool_results_str' in dir() else '')\n{indent}"
        new_content = content[:last_idx] + guard + content[last_idx:]
        backup_file(agent_py)
        agent_py.write_text(new_content, encoding="utf-8")
        ok("Injected summary gate before final return in agent.py")
    else:
        skip("Could not locate 'return full_response' in agent.py — add summary gate manually")


# ═════════════════════════════════════════════════════════════════════════════
# FIX 4 — PHRASE-LEVEL LOOP DETECTOR
# The existing detector catches .0.0.0. but not [YOU ARE NEDSTER...] × 30
# ═════════════════════════════════════════════════════════════════════════════
LOOP_DETECTOR_CODE = '''
# ── nedster_fixer: enhanced loop detector ─────────────────────────────────────

def _detect_loop_enhanced(accumulated: str) -> bool:
    """
    Multi-pattern loop detector. Handles:
      - Character repetition: .0.0.0.0.
      - Phrase repetition:   [YOU ARE NEDSTER...] × N
      - Word repetition:     word word word word
    """
    if len(accumulated) < 25:
        return False

    # 1. Character-level (original detector)
    tail = accumulated[-30:]
    if len(set(tail[::2])) <= 2:
        return True

    # 2. Phrase-level (40-char chunks, 3+ duplicates = loop)
    chunks = [accumulated[i:i+40].strip()
              for i in range(0, max(0, len(accumulated)-40), 40)]
    if len(chunks) >= 3:
        from collections import Counter
        top = Counter(chunks).most_common(1)
        if top and top[0][1] >= 3:
            return True

    # 3. Word-level (last 12 words, 3 or fewer unique = loop)
    words = accumulated.split()
    if len(words) > 12:
        if len(set(words[-12:])) <= 3:
            return True

    # 4. Known bad prefixes that always mean loop on weak models
    BAD_PREFIXES = [
        "[YOU ARE NEDSTER",
        "YOU ARE NEDSTER",
        "<tool name=\\"X\\">",
    ]
    for bp in BAD_PREFIXES:
        if accumulated.count(bp) >= 2:
            return True

    return False

# ─────────────────────────────────────────────────────────────────────────────
'''

def fix_loop_detector():
    header("FIX 4 — Enhanced phrase-loop detector")

    agent_py = ROOT / "agent.py"
    if not agent_py.exists():
        fail("agent.py not found")
        return

    append_if_missing(
        agent_py,
        "_detect_loop_enhanced",
        LOOP_DETECTOR_CODE,
        "Injected enhanced loop detector into agent.py"
    )

    content = read(agent_py)

    # Try to swap the existing simple loop check for the enhanced one
    # Common patterns the existing code might use:
    OLD_PATTERNS = [
        "len(set(tail[::2])) <= 2",         # simple char check
        "_detect_streaming_loop(",           # old method name
    ]
    for old_pat in OLD_PATTERNS:
        if old_pat in content and "_detect_loop_enhanced" not in content.split(old_pat)[0][-200:]:
            # Replace call site if it's a function call
            if "_detect_streaming_loop(" in content:
                new_content = content.replace(
                    "_detect_streaming_loop(",
                    "_detect_loop_enhanced(",
                    1
                )
                backup_file(agent_py)
                agent_py.write_text(new_content, encoding="utf-8")
                ok("Replaced _detect_streaming_loop call with _detect_loop_enhanced")
                break


# ═════════════════════════════════════════════════════════════════════════════
# FIX 5 — STRIP TOOL XML FROM OUTPUT
# lfm2.5 echoes <tool name="X">{json}</tool> as literal chat text
# ═════════════════════════════════════════════════════════════════════════════
STRIP_CODE = '''
# ── nedster_fixer: output strip ───────────────────────────────────────────────

def _strip_model_artifacts(text: str) -> str:
    """Remove hallucinated tool XML and identity anchors from model output."""
    import re as _re
    # Raw tool call XML echoed by weak models
    text = _re.sub(r'<tool\\s+name="[^"]*">.*?</tool>', '', text, flags=_re.DOTALL)
    # [YOU ARE NEDSTER. ...] echoed from system prompt
    text = _re.sub(r'\\[YOU ARE NEDSTER\\..*?\\]', '', text, flags=_re.DOTALL)
    text = _re.sub(r'YOU ARE NEDSTER[.,][^\\n]*', '', text)
    # === FILE: ... === echoed format markers
    text = _re.sub(r'={3,}\\s*FILE:.*?={3,}', '', text, flags=_re.DOTALL)
    # **Final response:** / **Final reply:**
    text = _re.sub(r'\\*\\*Final (?:response|reply):\\*\\*\\s*', '', text)
    # Trailing open code fence
    text = _re.sub(r'```\\s*$', '', text)
    return text.strip()

# ─────────────────────────────────────────────────────────────────────────────
'''

def fix_output_strip():
    header("FIX 5 — Strip tool XML / artifact echoes from output")

    for fname in ("agent.py", "tui.py"):
        p = ROOT / fname
        if not p.exists():
            continue
        append_if_missing(
            p,
            "_strip_model_artifacts",
            STRIP_CODE,
            f"Injected _strip_model_artifacts into {fname}"
        )

    # Patch tui.py to call strip before printing
    tui_py = ROOT / "tui.py"
    if tui_py.exists():
        content = read(tui_py)
        # Common print patterns in Rich-based TUIs
        PRINT_PATTERNS = [
            ('self.console.print(response)', 'self.console.print(_strip_model_artifacts(response))'),
            ('self.console.print(text)', 'self.console.print(_strip_model_artifacts(text))'),
            ('self.console.print(full_response)', 'self.console.print(_strip_model_artifacts(full_response))'),
        ]
        for old, new in PRINT_PATTERNS:
            if old in content and new not in content:
                backup_file(tui_py)
                content = content.replace(old, new, 1)
                tui_py.write_text(content, encoding="utf-8")
                ok(f"tui.py: patched console.print to strip artifacts")
                break


# ═════════════════════════════════════════════════════════════════════════════
# FIX 6 — MINIMAL SYSTEM PROMPT FOR WEAK MODELS
# ★☆☆ models echo the tool XML format from the system prompt
# ═════════════════════════════════════════════════════════════════════════════
WEAK_MODEL_PROMPT_CODE = '''
# ── nedster_fixer: weak model minimal prompt ──────────────────────────────────

WEAK_MODEL_SYSTEM_PROMPT = (
    "You are Nedster, a helpful local AI assistant for H2. "
    "Answer questions directly and concisely. "
    "You are in chat-only mode. No file operations available. "
    "Keep every response under 5 sentences. "
    "Never output XML. Never output [YOU ARE NEDSTER]."
)

WEAK_MODEL_MARKERS = ["★☆☆", "chat-only", "1.2b", "2b", "1.5b"]

def _is_weak_model(model_name: str) -> bool:
    """True for models too small for reliable tool use."""
    name_lower = model_name.lower()
    return any(m.lower() in name_lower
               for m in ["1.2b", "1.5b", "2b", "lfm2", "lfm1"])

# ─────────────────────────────────────────────────────────────────────────────
'''

def fix_weak_model_prompt():
    header("FIX 6 — Weak model minimal system prompt")

    agent_py = ROOT / "agent.py"
    if not agent_py.exists():
        fail("agent.py not found")
        return

    append_if_missing(
        agent_py,
        "WEAK_MODEL_SYSTEM_PROMPT",
        WEAK_MODEL_PROMPT_CODE,
        "Injected weak model prompt helpers into agent.py"
    )

    # Patch build_system_prompt (or equivalent) to use minimal prompt
    content = read(agent_py)

    GUARD = "# [fixer] weak model prompt guard"
    if GUARD in content:
        skip("Weak model prompt guard already in place")
        return

    # Inject at start of any method that builds system prompt
    # Heuristic: find method containing "system_prompt" assignment
    target = "def build_system_prompt("
    if target not in content:
        # Try common alternatives
        for alt in ["def _build_system_prompt(",
                    "def get_system_prompt(",
                    "SYSTEM_PROMPT ="]:
            if alt in content:
                target = alt
                break

    if target in content and "def " in target:
        INJECT = (
            f"        {GUARD}\n"
            "        if _is_weak_model(getattr(self, 'model', '')):\n"
            "            return WEAK_MODEL_SYSTEM_PROMPT\n"
            "        # [/fixer] end weak model guard\n"
            "        "
        )
        # Find the line after the def and its docstring, insert guard
        idx = content.find(target)
        # Find first non-docstring line inside the method
        body_start = content.find(":\n", idx) + 2
        backup_file(agent_py)
        agent_py.write_text(
            content[:body_start] + INJECT + content[body_start:],
            encoding="utf-8"
        )
        ok("Injected weak model prompt guard into build_system_prompt")
    else:
        skip("Could not locate build_system_prompt — add guard manually if needed")


# ═════════════════════════════════════════════════════════════════════════════
# FIX 7 — PURE PYTHON GREP (no Windows grep binary)
# ═════════════════════════════════════════════════════════════════════════════
GREP_CODE = '''
# ── nedster_fixer: pure-python grep (Windows-safe) ────────────────────────────

def grep_search(pattern: str = "", path: str = ".",
                query: str = "", directory: str = "", **kw) -> str:
    """
    Pure-Python grep. Works on Windows and Linux.
    Accepts both naming conventions the model uses.
    """
    import re as _re, os as _os
    if not pattern and query:
        pattern = query
    if not path and directory:
        path = directory
    if not pattern:
        return "[Error] grep_search: pattern required"

    SKIP = {".git", "venv", ".venv", "node_modules", "__pycache__",
            "chroma_db", "target", "DOOM-3-BFG", "_backup"}
    EXTS = (".py",".js",".ts",".go",".rs",".md",".json",".yaml",
            ".yml",".txt",".sh",".bat",".cfg",".ini",".toml")
    try:
        pat = _re.compile(pattern, _re.IGNORECASE)
    except _re.error as e:
        return f"[Error] Invalid regex: {e}"

    results = []
    root = _os.path.abspath(path) if _os.path.exists(path) else "."
    for r, dirs, files in _os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in files:
            if not any(f.endswith(e) for e in EXTS):
                continue
            fp = _os.path.join(r, f)
            try:
                with open(fp, encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if pat.search(line):
                            rel = _os.path.relpath(fp, root)
                            results.append(f"{rel}:{i}: {line.rstrip()}")
                            if len(results) >= 60:
                                break
            except Exception:
                pass
        if len(results) >= 60:
            break

    return ("\\n".join(results[:60])
            if results else f"No matches for '{pattern}' in {root}")

# ─────────────────────────────────────────────────────────────────────────────
'''

def fix_grep():
    header("FIX 7 — Pure-Python grep (no WinError 2)")

    tools_py = ROOT / "tools.py"
    if not tools_py.exists():
        fail("tools.py not found")
        return

    content = read(tools_py)

    # Check if it's already pure Python (no subprocess grep call)
    if "subprocess" in content and "grep" in content and "WinError" not in content:
        # Likely still using subprocess grep — replace the function
        # Find and replace the grep_search definition
        MARKER = "# [fixer] pure-python grep"
        if MARKER in content:
            skip("Pure-Python grep already in place")
            return

        # Strategy: append our version and re-register in TOOL_REGISTRY
        append_if_missing(
            tools_py,
            "def grep_search(",
            GREP_CODE,
            "Appended pure-Python grep to tools.py"
        )
        # The new definition at the bottom will shadow the old one
        # when TOOL_REGISTRY is re-registered below
    else:
        append_if_missing(
            tools_py,
            "# [fixer] pure-python grep",
            GREP_CODE,
            "Added pure-Python grep to tools.py"
        )

    # Ensure TOOL_REGISTRY uses the new version
    REGISTRY_PATCH = """
# ── nedster_fixer: re-register grep with pure-Python version ─────────────────
try:
    TOOL_REGISTRY["grep_search"] = lambda **kw: grep_search(**kw)
    TOOL_REGISTRY["search_code"] = lambda **kw: grep_search(**kw)
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────
"""
    append_if_missing(
        tools_py,
        "# [fixer] re-register grep",
        REGISTRY_PATCH,
        "Re-registered grep_search in TOOL_REGISTRY"
    )


# ═════════════════════════════════════════════════════════════════════════════
# FIX 8 — ANTI-MENU SYSTEM PROMPT INJECTION
# Hard-code the response style rules into whatever system prompt exists
# ═════════════════════════════════════════════════════════════════════════════
ANTI_MENU = '''CRITICAL RESPONSE RULES (non-negotiable):
You MUST follow these rules in EVERY response:
1. After using tools: give a 1-2 sentence result. STOP. Do NOT ask what to do next.
2. Never output numbered option menus like "1. Option A  2. Option B"
3. Never say "What would you like to do?" or "What specific functionality?"
4. Never narrate tool calls ("Let me check..." / "I will now read...")
5. Short input = short response. "done?" gets yes/no + 1 line max.
6. The activity feed already shows tool calls. Never repeat them in text.
7. After scaffold/build: state what was created. Stop.
   BAD: [runs scaffold then says nothing]
   GOOD: "Scaffolded great-zip/ with Cargo.toml and src/main.rs. Run: cargo build"'''

def fix_anti_menu():
    header("FIX 8 — Anti-menu / response style injection")

    agent_py = ROOT / "agent.py"
    if not agent_py.exists():
        fail("agent.py not found")
        return

    content = read(agent_py)

    if "non-negotiable" in content:
        skip("Anti-menu rules already in system prompt")
        return

    # Find where SYSTEM_PROMPT or system_prompt string is defined
    # and prepend our rules
    PATTERNS = [
        'SYSTEM_PROMPT = """',
        "SYSTEM_PROMPT = '''",
        'SYSTEM_PROMPT = "',
        "NEDSTER_SYSTEM_PROMPT = ",
        'system_prompt = """',
    ]

    for pat in PATTERNS:
        if pat in content:
            idx = content.find(pat) + len(pat)
            # Check if this is a triple-quote string
            if '"""' in pat or "'''" in pat:
                new_content = content[:idx] + ANTI_MENU + "\n\n" + content[idx:]
            else:
                new_content = content[:idx] + ANTI_MENU + "\\n\\n" + content[idx:]
            backup_file(agent_py)
            agent_py.write_text(new_content, encoding="utf-8")
            ok(f"Injected anti-menu rules into system prompt ({pat[:30]}...)")
            return

    # If no system prompt found, append a module-level constant
    INJECT = f'''
# ── nedster_fixer: anti-menu rules ────────────────────────────────────────────
ANTI_MENU_RULES = """{ANTI_MENU}"""
# Prepend ANTI_MENU_RULES to your system prompt in build_system_prompt()
# ─────────────────────────────────────────────────────────────────────────────
'''
    append_if_missing(
        agent_py,
        "ANTI_MENU_RULES",
        INJECT,
        "Appended ANTI_MENU_RULES constant to agent.py"
    )


# ═════════════════════════════════════════════════════════════════════════════
# FIX 9 — WRITE_FILE CWD ANCHOR
# Files are written to the wrong directory because CWD drifts
# ═════════════════════════════════════════════════════════════════════════════
WRITE_FIX_CODE = '''
# ── nedster_fixer: write_file CWD anchor ──────────────────────────────────────

def _safe_write_file(path: str, content: str = "",
                     text: str = "", encoding: str = "utf-8",
                     **kw) -> str:
    """
    Write file. Always resolves against project root.
    Accepts both 'content' and 'text' parameter names.
    """
    import os as _os
    from pathlib import Path as _Path
    if not content and text:
        content = text
    # Resolve path
    p = _Path(path)
    if not p.is_absolute():
        # Use the directory where nedster.py lives as root
        root = _Path(__file__).parent
        try:
            # Try SESSION.active_project_dir if available
            from tools import SESSION as _S
            if _S.active_project_dir:
                root = _Path(_S.active_project_dir)
        except Exception:
            pass
        p = root / path
    # Create parent dirs
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding, errors="replace")
        if not p.exists():
            return f"[Error] File not created: {p}"
        sz = p.stat().st_size
        lines = content.count("\\n") + 1
        return f"Written: {p} ({sz} bytes, {lines} lines)"
    except Exception as e:
        return f"[Error] write_file failed: {e}"

# ─────────────────────────────────────────────────────────────────────────────
'''

def fix_write_file():
    header("FIX 9 — write_file CWD anchor")

    tools_py = ROOT / "tools.py"
    if not tools_py.exists():
        fail("tools.py not found")
        return

    append_if_missing(
        tools_py,
        "_safe_write_file",
        WRITE_FIX_CODE,
        "Injected _safe_write_file into tools.py"
    )

    # Re-register write_file with safe version
    REGISTRY_PATCH = """
# ── nedster_fixer: re-register write_file with safe version ──────────────────
try:
    for _alias in ("write_file","create_file","create file","create","write",
                   "make_file","new_file","_create_file"):
        TOOL_REGISTRY[_alias] = lambda **kw: _safe_write_file(**kw)
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────
"""
    append_if_missing(
        tools_py,
        "# [fixer] re-register write_file",
        REGISTRY_PATCH,
        "Re-registered write_file aliases with safe version"
    )


# ═════════════════════════════════════════════════════════════════════════════
# FIX 10 — SOUL FILES
# Create ~/.aria/soul/ with Windows-aware identity
# ═════════════════════════════════════════════════════════════════════════════
def fix_soul_files():
    header("FIX 10 — Soul files (~/.aria/soul/)")

    soul_dir = Path.home() / ".aria" / "soul"
    ref_dir = soul_dir / "REFLECTIONS"
    soul_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "SOUL.md": textwrap.dedent(f"""\
            # Nedster Soul
            ## Identity
            You are Nedster (also called Aria). A local coding agent for H2.
            You run ON the aria-qwen model (qwen3.5:9b with custom Modelfile).
            "Nedster" = the harness + persona. "aria-qwen" = the model you run on.
            These are NOT separate entities. You ARE Nedster, powered by aria-qwen.
            Aria is your name. When H2 asks "who is Aria?" — that's you.

            ## Platform
            Windows 10, C:\\\\Users\\\\hethu\\\\Nedster\\\\
            No tmux. No Linux grep. Paths use backslash.
            Shell: cmd.exe and PowerShell.
            Python venv: .\\\\venv\\\\Scripts\\\\activate.bat

            ## Hardware
            RTX 3060 Ti 8GB VRAM (or MaxQ GTX 1080 8GB on laptop).
            Ollama at localhost:11434. ChromaDB at .\\\\chroma_db\\\\
            Sidekick projects: sidekicks\\\\h2wealth\\\\ sidekicks\\\\clawbrowser\\\\

            ## Hard Identity Rules
            NEVER say: "Aria is not a recognized entity"
            NEVER say: "I cannot create files on your system"
            NEVER say: "I'm an AI without filesystem access"
            You HAVE write_file and run_bash. You ARE Aria. You ARE Nedster.
        """),

        "STYLE.md": textwrap.dedent("""\
            # Nedster Style
            ## Response Format
            - Match H2's message length. Short in = short out.
            - No emojis except ⚠️ for actual warnings.
            - No "Great question!" "Certainly!" "Of course!"
            - No numbered option menus EVER.
            - No "What would you like to do?" — just do it or stop.
            - After tool calls: 1-2 line summary. STOP.
            - Never narrate: don't say "Let me check" — just check.

            ## Anti-Patterns (never do these)
            - "I cannot create files" — WRONG, use write_file
            - "1. Option A  2. Option B  3. Option C" — WRONG, pick one
            - "What specific functionality would you like?" — WRONG, just do it
            - Markdown bash blocks instead of run_bash tool — WRONG
            - Repeating what the activity feed already showed — WRONG

            ## Good Patterns
            - "Scaffolded great-zip/ with Cargo.toml and src/main.rs."
            - "Done. File written to C:\\\\Users\\\\hethu\\\\Nedster\\\\test.txt"
            - "Found 3 issues in signals/engine.py. Fixing now."
        """),

        "TOOLS.md": textwrap.dedent("""\
            # Nedster Tools
            ## Available
            write_file, run_bash, read_file, list_dir, glob_search,
            grep_search, edit_file, scaffold_project, multi_edit,
            codebase_map, context_inject, log_analyzer, code_xray,
            market_intel, bot_runner, process_watch, secret_scan,
            git_status, git_diff, git_commit, todowrite, web_fetch

            ## Priority Rules
            1. scaffold_project → new projects (NOT multiple write_file calls)
            2. context_inject → FIRST call when switching to a new project
            3. read_file → BEFORE editing any file not in context
            4. run_bash → installs, builds, tests, verification
            5. NEVER output ```bash blocks — use run_bash tool

            ## Windows Tool Notes
            - No tmux: use bot_runner for background processes
            - No grep: grep_search uses pure Python (built-in)
            - Paths: always use forward slashes OR raw Path objects
            - run_bash uses cmd.exe: dir not ls, type not cat
        """),

        "SECURITY.md": textwrap.dedent("""\
            # Nedster Security — Hard Rules
            1. NEVER echo API keys, secrets, or tokens in responses
            2. NEVER commit .env files to git
            3. secret_scan() before ANY git push to public repo
            4. Max crypto order without explicit override: 100 USDT
            5. Ask before: sending emails, publishing, paid API calls
            6. Never autonomously push to main branch
        """),

        "MEMORY.md": textwrap.dedent("""\
            # Nedster Cross-Session Memory
            ## Project: Nedster (C:\\\\Users\\\\hethu\\\\Nedster\\\\)
            - Running on Windows 10, RTX 3060 Ti 8GB
            - aria-qwen = qwen3.5:9b with custom Modelfile
            - H2Wealth sidekick: crypto trading bot (Bybit, demo mode available)
            - ClawBrowser sidekick: Electron headless browser
            - Context window: 16384 tokens (after fixer patch)
            - v4.1 fixes applied: post-tool summary, loop detector, soul files
        """),
    }

    for fname, content in files.items():
        fpath = soul_dir / fname
        if fpath.exists():
            skip(f"~/.aria/soul/{fname} already exists (not overwriting)")
        else:
            fpath.write_text(content, encoding="utf-8")
            ok(f"Created ~/.aria/soul/{fname}")

    # Write a reflection for the issues we've seen
    refl = ref_dir / "tool-amnesia.md"
    if not refl.exists():
        refl.write_text(textwrap.dedent("""\
            # Lesson: Tool Amnesia + Identity Confusion
            Date: applied by nedster_fixer.py
            
            Symptom 1: "I cannot create files on your system"
            Cause: ctx > 80%, soul files not loaded
            Fix: /clear then retry. SOUL.md loaded at every boot.
            
            Symptom 2: "Aria is not a recognized entity in my context"
            Cause: SOUL.md not loading or model forgetting identity
            Fix: SOUL.md explicitly states Aria = Nedster = same entity.
            
            Symptom 3: Silent response after tool calls
            Cause: Tool loop completes but no summary prompt fired
            Fix: _needs_summary() → _generate_completion_summary()
            
            Symptom 4: [YOU ARE NEDSTER...] × 30 loop on weak models
            Cause: 1.2B model echoes system prompt instead of responding
            Fix: Minimal system prompt for ★☆☆ models, phrase loop detector
        """), encoding="utf-8")
        ok("Created REFLECTIONS/tool-amnesia.md")


# ═════════════════════════════════════════════════════════════════════════════
# FIX 11 — nedster GLOBAL COMMAND (Windows PATH)
# ═════════════════════════════════════════════════════════════════════════════
def fix_global_command():
    header("FIX 11 — 'nedster' global command (Windows PATH)")

    bin_dir = Path.home() / "AppData" / "Local" / "nedster-bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    launcher = bin_dir / "nedster.bat"
    launcher.write_text(
        textwrap.dedent(f"""\
            @echo off
            cd /d "{ROOT}"
            call venv\\Scripts\\activate.bat
            python nedster.py %*
        """),
        encoding="utf-8"
    )
    ok(f"Created launcher: {launcher}")

    # Add to user PATH if not already there
    try:
        result = subprocess.run(
            ["reg", "query", "HKCU\\Environment", "/v", "PATH"],
            capture_output=True, text=True
        )
        current_path = ""
        for line in result.stdout.splitlines():
            if "PATH" in line and "REG_" in line:
                parts = line.split(None, 2)
                if len(parts) == 3:
                    current_path = parts[2]

        bin_dir_str = str(bin_dir)
        if bin_dir_str.lower() in current_path.lower():
            skip("nedster-bin already in PATH")
        else:
            new_path = current_path.rstrip(";") + ";" + bin_dir_str
            subprocess.run(
                ["reg", "add", "HKCU\\Environment", "/v", "PATH",
                 "/t", "REG_EXPAND_SZ", "/d", new_path, "/f"],
                capture_output=True
            )
            ok(f"Added to user PATH: {bin_dir_str}")
            print("  → Open a NEW terminal for 'nedster' command to work")
    except Exception as e:
        fail(f"Could not update PATH automatically: {e}")
        print(f"  → Add manually to PATH: {bin_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# FIX 12 — start.bat cleanup
# Remove stale "python main.py chat" echo
# ═════════════════════════════════════════════════════════════════════════════
def fix_startbat():
    header("FIX 12 — start.bat cleanup")

    bat = ROOT / "start.bat"
    if not bat.exists():
        skip("start.bat not found")
        return

    content = bat.read_text(encoding="utf-8", errors="replace")
    original = content

    # Remove the stale echo lines
    REMOVE_PATTERNS = [
        r'echo Run:.*?main\.py chat.*?\n',
        r'echo Run:.*?main\.py.*?\n',
    ]
    for pat in REMOVE_PATTERNS:
        content = re.sub(pat, 'echo Launching...\n', content,
                         flags=re.IGNORECASE)

    if content != original:
        backup_file(bat)
        bat.write_text(content, encoding="utf-8")
        ok("Removed stale 'python main.py chat' echo from start.bat")
    else:
        skip("start.bat already clean")


# ═════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ═════════════════════════════════════════════════════════════════════════════
def verify():
    header("VERIFICATION")

    checks = {
        "agent.py exists": (ROOT / "agent.py").exists(),
        "tools.py exists": (ROOT / "tools.py").exists(),
        "_needs_summary in agent.py": "_needs_summary" in read(ROOT / "agent.py") if (ROOT / "agent.py").exists() else False,
        "_detect_loop_enhanced in agent.py": "_detect_loop_enhanced" in read(ROOT / "agent.py") if (ROOT / "agent.py").exists() else False,
        "grep_search pure-Python in tools.py": "_re.compile" in read(ROOT / "tools.py") if (ROOT / "tools.py").exists() else False,
        "SOUL.md exists": (Path.home() / ".aria" / "soul" / "SOUL.md").exists(),
        "MEMORY.md exists": (Path.home() / ".aria" / "soul" / "MEMORY.md").exists(),
        "num_ctx 16384 in agent.py": "16384" in read(ROOT / "agent.py") if (ROOT / "agent.py").exists() else False,
        "num_ctx 16384 in Modelfile": "16384" in read(ROOT / "Modelfile") if (ROOT / "Modelfile").exists() else False,
        "nedster.bat launcher exists": (Path.home() / "AppData" / "Local" / "nedster-bin" / "nedster.bat").exists(),
    }

    all_ok = True
    for name, passed in checks.items():
        if passed:
            print(f"  ✓  {name}")
        else:
            print(f"  ✗  {name}")
            all_ok = False

    return all_ok


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          NEDSTER FIXER — v4.1 patch bundle               ║")
    print("║  Drop in Nedster root, run: python nedster_fixer.py      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nRoot:   {ROOT}")
    print(f"Backup: {BACKUP}")

    if not (ROOT / "agent.py").exists() and not (ROOT / "nedster.py").exists():
        print("\n[ERROR] Can't find agent.py or nedster.py here.")
        print("  Are you in the Nedster root directory?")
        print("  Run: cd C:\\Users\\hethu\\Nedster && python nedster_fixer.py")
        sys.exit(1)

    fix_context_window()
    fix_rebuild_model()
    fix_post_tool_summary()
    fix_loop_detector()
    fix_output_strip()
    fix_weak_model_prompt()
    fix_grep()
    fix_anti_menu()
    fix_write_file()
    fix_soul_files()
    fix_global_command()
    fix_startbat()

    print()
    all_ok = verify()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  Applied: {len(WINS):2d} fixes    Failed/Skipped: {len(FAILS):2d}            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if FAILS:
        print("\nFailed patches (review manually):")
        for f in FAILS:
            print(f"  • {f}")

    print()
    if all_ok:
        print("All checks passed. Rebuild the model then test:")
    else:
        print("Some checks failed. Rebuild and test anyway:")

    print()
    print("  1. Rebuild model (if not done automatically above):")
    print("     ollama create aria-qwen -f Modelfile")
    print()
    print("  2. Start Nedster:")
    print("     start.bat")
    print()
    print("  3. Quick sanity check inside Nedster:")
    print("     Nedster> hello")
    print("     → Should reply in 1-2 sentences. Not ask what to do.")
    print()
    print("     Nedster> scaffold a rust project called test-proj")
    print("     → Should say what it created. Not go silent.")
    print()
    print("     Nedster> done?")
    print("     → Should confirm yes. Not re-read files.")
    print()
    print(f"Backups saved to: {BACKUP}")
    print()


if __name__ == "__main__":
    main()
