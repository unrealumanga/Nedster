
class ToolExecutor:
    """
    Decoupled tool execution interface.
    The harness calls tools; the brain sees only tool results.
    Pattern: Anthropic managed agents brain/hands split.
    """
    def __init__(self, registry: dict, auto: bool = False, session_log=None):
        self.registry = registry
        self.auto = auto
        self.session_log = session_log
        self._budget_remaining = 50
        self._call_count = 0

    def execute(self, name: str, args: dict, tui=None) -> str:
        self._call_count += 1
        self._budget_remaining -= 1
        if self.session_log:
            self.session_log.emit("tool_call", {"tool": name, "args": args, "call_n": self._call_count})

        if name not in self.registry:
            return f"[ERROR: '{name}' unknown. Use write_file to create files.]"

        try:
            result = self.registry[name](**args)
            result = str(result)
        except Exception as e:
            result = f"[ERROR] {name} failed: {e}"

        if self.session_log:
            self.session_log.emit("tool_result", {"tool": name, "result": result[:200]})
        return result

    def budget_exhausted(self) -> bool:
        return self._budget_remaining <= 0

import os

"""Nedster Agent - Core agentic loop extending RAG pipeline"""

import re
import ollama
from datetime import datetime
from typing import Optional, Dict, List

from context_loader import ContextLoader
from editor import FileEditor
from rag_engine.retriever import Retriever
from memory import MemoryManager
from tools import TOOL_REGISTRY, parse_tool_calls, WATCHDOG
from tui import NedsterTUI


def _build_verification_injection(tool_results_str: str) -> str:
    """
    Scan tool results for errors/warnings.
    Build an explicit verification directive for the model.
    """
    import re

    has_error = bool(
        re.search(
            r"ERROR|error|No such file|not found|failed|WARN.*Unknown tool",
            tool_results_str,
            re.IGNORECASE,
        )
    )

    has_success = bool(
        re.search(
            r"Written:|Created:|bytes\)|Successfully|PONG|total \d+", tool_results_str
        )
    )

    if has_error and not has_success:
        return (
            "\n[EXECUTION RESULT: ALL TOOLS FAILED]\n"
            "The tool results above contain only errors.\n"
            "DO NOT report success. DO NOT say files were created.\n"
            "State what failed and why. Then try a different approach.\n"
            "If 'Unknown tool': use run_bash to create files instead.\n"
        )
    elif has_error:
        return (
            "\n[EXECUTION RESULT: PARTIAL FAILURE]\n"
            "Some tools succeeded, some failed (see errors above).\n"
            "Report only what ACTUALLY succeeded based on tool output.\n"
            "Do not claim success for the failed operations.\n"
        )
    else:
        return (
            "\n[EXECUTION RESULT: SUCCESS]\n"
            "Report results based on actual tool output above.\n"
            "Include file paths and sizes from the tool results.\n"
        )


def _verify_task_completion(user_input: str, tool_results: str) -> str | None:
    """
    For tasks that create files/dirs, verify they actually exist.
    Returns a warning string if verification fails, None if OK.
    """
    import os, re

    # Detect "create project/bot/folder" intent
    CREATE_INTENT = re.compile(
        r"create|build|make|scaffold|new\s+(?:bot|project|folder|app)", re.IGNORECASE
    )
    if not CREATE_INTENT.search(user_input):
        return None

    # Extract paths mentioned in tool results
    PATH_RE = re.compile(r"(/home/\S+|~/\S+)")
    paths_mentioned = PATH_RE.findall(tool_results)

    for path in paths_mentioned:
        path = os.path.expanduser(path.rstrip("/.,)"))
        if os.path.isdir(path):
            files = os.listdir(path)
            if not files:
                return (
                    f"\n⚠️ TASK INCOMPLETE: {path} was created "
                    f"but is EMPTY. Files were not written.\n"
                    f"Use scaffold_project to create all files atomically."
                )
            elif len(files) < 3:
                return (
                    f"\n⚠️ TASK PARTIAL: {path} has only "
                    f"{len(files)} file(s): {files}. "
                    f"Expected more for a complete project.\n"
                )
    return None


TOOL_REFUSAL_PATTERNS = [
    r"I cannot (?:directly )?create files",
    r"I don't have (?:direct )?(?:filesystem|file system|shell|file) (?:access|tools)",
    r"I'?m an AI (?:assistant )?without",
    r"I can'?t (?:actually )?(?:create|write|access|execute)",
    r"I don'?t have the ability to",
    r"without (?:direct )?(?:API|filesystem) access",
    r"I'?m unable to (?:directly )?(?:create|write|access)",
    r"I don'?t have (?:direct )?access to your",
    r"as an AI(?:,| I)",
    r"I cannot directly read or write files",
    r"I don't have direct file system tools",
    r"not a local agent",
    r"If you have a shell tool available",
    r"You can create the file manually"
]


def _detect_tool_refusal(response: str) -> bool:
    for pat in TOOL_REFUSAL_PATTERNS:
        if re.search(pat, response, re.IGNORECASE):
            return True
    return False



class IterationBudget:
    def __init__(self, max_iters: int = 10, max_chars: int = 12000):
        self.max_iters = max_iters
        self.max_chars = max_chars
        self._iters = 0
        self._chars = 0
    
    @property
    def remaining(self) -> int:
        return max(0, self.max_iters - self._iters)
    
    def consume(self, messages: list) -> bool:
        self._iters += 1
        self._chars = sum(len(m.get("content","")) for m in messages)
        return (self._iters <= self.max_iters and self._chars <= self.max_chars)
    
    def inject_limit_message(self) -> str:
        if self._iters >= self.max_iters:
            return "\n[TOOL LIMIT: Stop calling tools. Summarize what you found and answer directly.]"
        return "\n[CONTEXT LIMIT: Stop calling tools. Summarize what you found and answer directly.]"

class PluginHooks:
    def __init__(self):
        self._hooks = {
            "pre_llm_call": [],
            "post_llm_call": [],
            "on_session_start": [],
            "on_session_end": [],
            "on_tool_result": [],
        }
        self._load_plugins()
    
    def _load_plugins(self):
        import os, importlib.util, glob
        plugin_dir = os.path.expanduser("~/.aria/plugins")
        os.makedirs(plugin_dir, exist_ok=True)
        for path in glob.glob(os.path.join(plugin_dir, "*.py")):
            try:
                spec = importlib.util.spec_from_file_location("plugin", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for hook in self._hooks:
                    if hasattr(mod, hook):
                        self._hooks[hook].append(getattr(mod, hook))
            except Exception as e:
                print(f"[Plugin] {path}: {e}")
    
    def fire(self, hook: str, **kwargs):
        for fn in self._hooks.get(hook, []):
            try: fn(**kwargs)
            except Exception: pass


class NedsterAgent:

    def _update_state(self, key: str, value):
        import json
        from pathlib import Path
        from datetime import datetime
        state_path = Path(self.project_dir) / "nedster_state.json"
        try:
            with open(state_path) as f: state = json.load(f)
        except Exception:
            state = {
                "project": str(self.project_dir),
                "model": self.model,
                "created": datetime.now().isoformat(),
                "tasks": [], "artifacts": [], "milestones": [], "context": {}
            }
        
        if key == "task":
            existing = next((t for t in state["tasks"] if t.get("id") == value.get("id")), None)
            if existing: existing.update(value)
            else: state["tasks"].append(value)
        elif key == "artifact":
            state["artifacts"].append({**value, "created": datetime.now().isoformat()})
        elif key == "milestone":
            state["milestones"].append({"text": value, "ts": datetime.now().isoformat()})
        else:
            state["context"][key] = value

        state["updated"] = datetime.now().isoformat()
        with open(state_path, "w") as f: json.dump(state, f, indent=2)

    def _read_state(self) -> dict:
        import json
        from pathlib import Path
        state_path = Path(self.project_dir) / "nedster_state.json"
        try:
            with open(state_path) as f: return json.load(f)
        except Exception:
            ned_path = Path(self.project_dir) / "NEDSTER.md"
            if ned_path.exists():
                with open(ned_path) as f:
                    return {"context": {"nedster_md": f.read()}}
            return {}

    """
    NedsterAgent - Local coding agent with RAG, context awareness, and tool use.
    Extends the Aria RAG pipeline with project context and code editing.
    """


    def build_system_prompt(self, tool_use_enabled: bool = True, model_tier: str = "★★★") -> str:
        # [fixer] weak model prompt guard
        if _is_weak_model(getattr(self, 'model', '')):
            return WEAK_MODEL_SYSTEM_PROMPT
        # [/fixer] end weak model guard
        if model_tier == "★☆☆":
            return (
                "You are Nedster, a helpful local AI assistant for H2. "
                "Answer questions directly and concisely. "
                "You are in chat-only mode — no file operations. "
                "Keep responses under 5 sentences. "
                "Never output XML tags. Never output [YOU ARE NEDSTER]."
            )
        
        soul_context = self._load_soul_files()
        system_prompt = soul_context + "\n\n" + self.NEDSTER_SYSTEM_PROMPT
        
        from tools import SESSION
        PLATFORM_INJECTION = f"""
Platform: {SESSION.platform}
Shell: {"cmd.exe / PowerShell" if SESSION.platform == "windows" else "bash"}
"""
        system_prompt += "\n" + PLATFORM_INJECTION
        
        ANTI_MENU_INJECTION = """
RESPONSE RULES — ABSOLUTE:
1. After using tools: give a 1-2 sentence result summary. STOP.
   Bad:  [runs scaffold_project then says nothing]
   Good: "Scaffolded great-zip/ with Cargo.toml and src/main.rs."
2. NEVER output "What would you like to do?" — just stop.
3. NEVER show numbered menus like "1. Read  2. Modify  3. Add"
4. NEVER narrate tool use: don't say "Let me check..." — just check.
5. Short user input = short response. "done?" = yes/no + 1 line.
6. The activity feed shows tool calls. Don't repeat them in text.
"""
        system_prompt = ANTI_MENU_INJECTION + "\n" + system_prompt
        
        return system_prompt

    NEDSTER_SYSTEM_PROMPT = """CRITICAL RESPONSE RULES (non-negotiable):
You MUST follow these rules in EVERY response:
1. After using tools: give a 1-2 sentence result. STOP. Do NOT ask what to do next.
2. Never output numbered option menus like "1. Option A  2. Option B"
3. Never say "What would you like to do?" or "What specific functionality?"
4. Never narrate tool calls ("Let me check..." / "I will now read...")
5. Short input = short response. "done?" gets yes/no + 1 line max.
6. The activity feed already shows tool calls. Never repeat them in text.
7. After scaffold/build: state what was created. Stop.
   BAD: [runs scaffold then says nothing]
   GOOD: "Scaffolded great-zip/ with Cargo.toml and src/main.rs. Run: cargo build"

DIRECTIVE ZERO — LANGUAGE:
English only. Every word, every thought.

DIRECTIVE ONE — IDENTITY:
You are Nedster, a local coding agent for H2.
You operate on their codebase with these edit formats:
  <edit file="path"><old>...</old><new>...</new></edit>
  <create file="path">...</create>
Use these for ALL file changes - never describe changes, make them.

DIRECTIVE TWO — CODE EDITS:
When asked to fix/implement/refactor:
  1. Read the relevant file first (if not in context)
  2. Make the minimal edit - not a rewrite unless asked
  3. Always emit <edit> or <create> blocks - never "you should change X"
  4. After edit: check_syntax() automatically
  5. Never leave TODOs or placeholder comments

DIRECTIVE THREE — PLANNING:
For multi-file tasks, emit a plan first:
  Step 1: [what] in [file]
  Step 2: [what] in [file]
  Awaiting approval.
On any poke ("ok", "go", "yes", "!") - execute all steps.

DIRECTIVE FOUR — GIT:
After completing a coding task:
  - Run git_status() silently
  - If changes exist and H2 hasn't mentioned git: offer one commit line
  - Never commit without mentioning it

DIRECTIVE FIVE — TOOL USAGE:
To use a tool, use this EXACT XML format:
<tool name="tool_name">{"arg1": "value"}</tool>

Example:
<tool name="run_bash">{"cmd": "grep -rnw . -e 'TODO'"}</tool>
<tool name="read_file">{"path": "foo.py"}</tool>

TOOL PRIORITY:
read_file - before editing any file not in context
run_bash  - for installs, builds, verification
run_tests - after edits that affect logic
git_*     - after successful task completion

DIRECTIVE SIX — nedster_state.json:
On task completion, extract any project facts:
  architecture decisions, dependencies added, patterns used.
Append to nedster_state.json silently. Say: "Project memory updated."

DIRECTIVE SEVEN — HARDWARE:
H2 hardware: RTX 3060 Ti 8GB, i7-11700k, 64GB, Pop!OS.
aria-qwen context limit: 4096 tokens.
Keep context lean. Prioritize minimal file loading.

DIRECTIVE EIGHT — BEHAVIOR:
- No emojis except for warnings
- No "Great question!", "Certainly!", "Of course!"
- No "Would you like me to..." - just do it
- Match response length to input length
- Execute, don't narrate

[NON-NEGOTIABLE EXECUTION RULES]

RULE 1 — NEVER report success without tool confirmation:
  Wrong: "Files copied to X" (when tools returned errors)
  Right: Read the [Tool result:] output. Report THAT.

RULE 2 — If you see [WARN] Unknown tool or ERROR in results:
  Stop. Do not continue claiming success.
  Switch approach: use run_bash or scaffold_project instead.
  
RULE 3 — To create a new project with multiple files:
  ALWAYS use scaffold_project in ONE call.
  NEVER use multiple write_file calls for a new project.
  scaffold_project creates the dir, files, git init atomically.

RULE 4 — To verify your work:
  After creating files: call list_dir on the target directory.
  The list_dir result is TRUTH. Not your intention.
  If list_dir shows empty: the files were NOT created.

RULE 5 — run_bash runs in SESSION.active_project_dir:
  mkdir creates the dir in the active project directory.
  Write files using ABSOLUTE paths to guarantee location.

RULE 6 — You are NOT done until list_dir confirms files exist.

"""

    def __init__(self, project_dir: str, auto: bool = False, think: bool = False):
        from tools import SESSION

        SESSION.set_project(project_dir)

        import sys, platform
        from tools import SESSION
        SESSION.platform = "windows" if sys.platform == "win32" else "linux"
        SESSION.platform_note = "Windows 10 — no tmux, use tasklist/taskkill, paths use backslash" if SESSION.platform == "windows" else "Linux/Pop!OS — tmux available, bash available"

        self.project_dir = project_dir
        self.auto = auto
        self.think = think
        self.verbose = False
        self.model = os.environ.get("MODEL", "qwen3.5:9b")

        self.context_loader = ContextLoader(project_dir)
        self.editor = FileEditor(project_dir)
        self.retriever = Retriever()
        self.memory = MemoryManager(self.model)
        self.tui = NedsterTUI()
        self.hooks = PluginHooks()
        from memory import SessionLog
        from tools import TOOL_REGISTRY
        self.session_log = SessionLog(self.memory.session_id)
        self.executor = ToolExecutor(TOOL_REGISTRY, auto=self.auto, session_log=self.session_log)

        self.tool_stats = {"calls": 0, "loops": 0, "edits": 0, "tests": 0}
        self.pending_plan: Optional[str] = None
        self.plan_steps: List[str] = []
        self.current_step = 0

        self._boot_project()

    def _boot_project(self):
        """
        1. context_loader.scan_project()
        2. Read nedster_state.json - inject into system prompt
        3. Load milestones (existing logic)
        4. probe_tools() - tool inventory
        5. Print boot summary
        """
        # Scan project
        file_count = self.context_loader.scan_project()

        # Get vector count
        vector_count = 0
        try:
            import chromadb

            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_collection(name="rag_docs")
            vector_count = collection.count()
        except Exception:
            pass

        # Read nedster_state.json
        nedster_md = self.context_loader.read_nedster_md()

        # Print tool status
        from tools import probe_tools, check_model_available

        tool_status = probe_tools()
        tools_ok_list = []
        tools_warn_list = []
        for k, v in tool_status.items():
            if v == "OK":
                tools_ok_list.append(f"{k} ✓")
            else:
                tools_warn_list.append(f"{k} ✗")

        tools_ok_str = "  ".join(tools_ok_list)
        tools_warn_str = "  ".join(tools_warn_list) if tools_warn_list else ""

        session_count = 0
        path = os.path.expanduser("~/.aria/milestones.md")
        if os.path.exists(path):
            with open(path) as f:
                session_count = sum(1 for line in f if line.startswith("## Session "))

        vram_free = "0.0 GB"
        vram_total = "8.0 GB"
        try:
            import subprocess

            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            parts = r.stdout.strip().split(",")
            if len(parts) == 2:
                vram_free = f"{float(parts[0]) / 1024:.1f} GB"
                vram_total = f"{float(parts[1]) / 1024:.1f} GB"
        except Exception:
            pass

        info = check_model_available(self.model)
        model_size = "loading..."

        project_name = os.path.basename(os.path.abspath(self.project_dir)) or "project"
        self.tui.print_boot(
            project_name,
            file_count,
            vector_count,
            session_count,
            self.model,
            model_size,
            vram_free,
            vram_total,
            tools_ok_str,
            tools_warn_str,
            self.think,
            self.auto,
        )

        if nedster_md:
            self.tui.print_status("Project memory: nedster_state.json loaded")

        from daemon import read_pending_alerts
        alerts = read_pending_alerts()
        if alerts:
            urgent = [a for a in alerts if a.get("priority")=="urgent"]
            normal = [a for a in alerts if a.get("priority") != "urgent"]
            if urgent:
                for a in urgent: self.tui.print_warning(f"[{a['daemon'].upper()}] {a['message']}")
            if normal:
                self.tui.print_status(f"{len(normal)} daemon alerts (run /daemon to view)")

        # Ingest queue pickup
        from pathlib import Path
        import json
        queue_path = Path.home() / ".aria" / "ingest_queue.json"
        if queue_path.exists():
            try:
                with open(queue_path) as f:
                    queue = json.load(f)
                if queue:
                    self.tui.print_status(f"[file-watch] {len(queue)} files queued for RAG ingest — processing...")
                    for fpath in queue[:10]:
                        if os.path.exists(fpath):
                            try:
                                from rag_engine.ingestion import get_text_from_file, chunk_text, embed_file_chunks
                                import chromadb
                                client = chromadb.PersistentClient(path="./chroma_db")
                                coll = client.get_or_create_collection("rag_docs")
                                text = get_text_from_file(fpath)
                                if text:
                                    chunks = chunk_text(text, fpath)
                                    embed_file_chunks(chunks, self.retriever.embedder, coll, os.path.basename(fpath))
                            except Exception: pass
                    queue_path.unlink()
            except Exception: pass



    def _load_soul_files(self) -> str:
        import os, glob
        soul_dir = os.path.expanduser("~/.aria/soul")
        if not os.path.exists(soul_dir):
            os.makedirs(soul_dir)
            self._write_default_soul_files(soul_dir)
        
        parts = []
        for fname in ["SOUL.md", "STYLE.md", "TOOLS.md", "HEARTBEAT.md", "SECURITY.md"]:
            fpath = os.path.join(soul_dir, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    content = f.read().strip()
                if content:
                    parts.append(f"[{fname}]\n{content}")
        
        mem_path = os.path.join(soul_dir, "MEMORY.md")
        if os.path.exists(mem_path):
            with open(mem_path) as f:
                lines = f.readlines()
            recent = "".join(lines[-40:]).strip()
            if recent:
                parts.append(f"[MEMORY.md - recent]\n{recent}")
        
        ref_dir = os.path.join(soul_dir, "REFLECTIONS")
        if os.path.exists(ref_dir):
            ref_files = sorted(glob.glob(os.path.join(ref_dir, "*.md")), key=os.path.getmtime, reverse=True)[:2]
            for rf in ref_files:
                with open(rf) as f:
                    content = f.read().strip()
                if content:
                    fname = os.path.basename(rf)
                    parts.append(f"[REFLECTIONS/{fname}]\n{content[:300]}")
        
        combined = "\n\n".join(parts)
        if len(combined) > 3200:
            combined = combined[:3200] + "\n...[soul truncated]"
        return combined

    def _write_default_soul_files(self, soul_dir: str):
        import os
        os.makedirs(os.path.join(soul_dir, "REFLECTIONS"), exist_ok=True)
        
        files = {
            "SOUL.md": "# Nedster Soul\n## Identity\nYou are Nedster (codename: Aria). A local coding agent for H2.\nYou run ON the qwen3.5:9b model.\n\"Nedster\" = the harness + persona. \"qwen3.5:9b\" = the model you run on.\nThese are NOT different entities. You are Nedster, powered by qwen3.5:9b.\n\n## Platform\nRunning on: Windows 10, RTX 3060 Ti 8GB (or MaxQ GTX 1080 8GB laptop).\nTools run via Windows CMD/PowerShell — no tmux, no Linux grep, no /home paths.\nWindows paths use backslash: C:\\Users\\hethu\\Nedster\\\nHome dir: C:\\Users\\hethu\\\nProject root: wherever nedster.py is running from (os.getcwd()).\n\n## Hardware\nOllama at localhost:11434\nChromaDB at ./chroma_db\nPython venv at ./venv\n\n## Core Character\nSharp, direct, local. Execute — don't explain. Mirror H2's length.\nNEVER say \"I'm Nedster. Aria is a different model.\" — Aria IS Nedster.\nNEVER say \"Aira is not a recognized entity\" — Aria is your name.\nNEVER say \"I don't have filesystem access\" — you have write_file.",
            "STYLE.md": "# Nedster Style\n## Response Format\nMatch H2 message length exactly",
            "TOOLS.md": "# Nedster Tools\n## Available Tools...",
            "SECURITY.md": "# Nedster Security Rules\n## HARD RULES...",
            "HEARTBEAT.md": "# Nedster Heartbeat Behaviors\n## Daemon Check Intervals...",
        }
        for fname, content in files.items():
            fpath = os.path.join(soul_dir, fname)
            if not os.path.exists(fpath):
                with open(fpath, "w") as f: f.write(content)
        
        ref_path = os.path.join(soul_dir, "REFLECTIONS", "tool-amnesia.md")
        if not os.path.exists(ref_path):
            with open(ref_path, "w") as f:
                f.write("## 2026-04-12\nDrift detected. Recovering.")

    def _append_reflection(self, theme: str, content: str):
        import os
        from datetime import datetime
        ref_dir = os.path.expanduser("~/.aria/soul/REFLECTIONS")
        os.makedirs(ref_dir, exist_ok=True)
        path = os.path.join(ref_dir, f"{theme}.md")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{content}\n")

    def _is_complex_task(self, user_input: str) -> bool:
        COMPLEX_SIGNALS = ["build", "create a", "implement", "scaffold", "write a bot", "make a", "develop", "set up", "build me", "new project", "trading bot", "full stack", "microservice", "new folder"]
        lower = user_input.lower()
        return (any(s in lower for s in COMPLEX_SIGNALS) and len(user_input.split()) > 8)

    def _plan_phase(self, user_input: str) -> list:
        resp = ollama.generate(
            model=self.model,
            prompt=(
                f"Break this into 3-5 concrete steps. "
                f"Each step = one tool call action. Be specific.\n"
                f"Task: {user_input}\n\n"
                f"Format: 1. [tool] specific action\n"
                f"Output ONLY the numbered list. No preamble."
            ),
            options={"num_ctx": 16384, "num_predict": 200, "temperature": 0.1}
        )
        plan_text = resp["response"].strip()
        steps = [l.strip() for l in plan_text.split("\n") if l.strip() and l.strip()[0].isdigit()]
        if not steps: steps = [plan_text]
        todos = [{"id": str(i+1), "content": s, "status": "pending"} for i, s in enumerate(steps)]
        from tools import todowrite
        todowrite(todos)
        return steps

    def _verify_phase(self, steps: list, tool_results: str) -> str:
        import re, os
        checks = []
        PATH_RE = re.compile(r'(/home/[\w/\-_.]+\.\w+|~/[\w/\-_.]+\.\w+|[A-Za-z]:\\[\w/\\\-_.]+\.\w+)')
        for i, step in enumerate(steps):
            for path in PATH_RE.findall(step):
                expanded = os.path.expanduser(path)
                if os.path.exists(expanded):
                    checks.append(f"  ✓ {path}")
                    self._update_state("task", {
                        "id": str(i+1),
                        "status": "completed",
                        "verified": True,
                        "verify_method": "file_exists",
                        "verified_path": expanded
                    })
                elif any(s in tool_results for s in ["Written:", "Created:", "[exit 0]"]):
                    checks.append(f"  ~ {path} (in tool output)")
                else:
                    checks.append(f"  ✗ {path} — NOT found on disk")
        if not checks:
            return "Verify: no file paths to check."
        return "Verify:\n" + "\n".join(checks)

    def _get_model_size(self, model: str) -> float:
        import subprocess

        try:
            r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if model.lower() in line.lower():
                    m = re.search(r"(\d+\.?\d*)\s*GB", line)
                    if m:
                        return float(m.group(1))
        except Exception:
            pass
        return 0.0

    def generate(self, user_input: str, think: Optional[bool] = None):
        from tools import SESSION
        """
        Full agentic loop for a single user input.
        """
        think_enabled = think if think is not None else self.think

        if not think_enabled:
            lower_input = user_input.lower()
            THINK_TRIGGERS = [
                "architect",
                "design a system",
                "why is",
                "debug",
                "not working",
                "plan",
                "tradeoff",
                "compare",
                "should i",
                "best approach",
                "refactor entire",
            ]
            if any(t in lower_input for t in THINK_TRIGGERS):
                think_enabled = True
                self.tui.print_status("[Think: auto-enabled for this query]")

        try:
            # Phase 0 - AUTO-DETECT PROJECT
            PATH_RE = re.compile(r"(?:~/|/home/\w+/|\./)[\w/\-_.]+")
            for match in PATH_RE.finditer(user_input):
                path_str = match.group(0)
                expanded = os.path.expanduser(path_str)
                if os.path.isdir(expanded):
                    import pathlib

                    new_proj = pathlib.Path(expanded).resolve()
                    if str(new_proj) != str(self.project_dir):
                        # from tools import SESSION

                        SESSION.set_project(str(new_proj))
                        self.project_dir = str(new_proj)
                        self.context_loader.root = new_proj
                        self.editor.project_dir = new_proj
                        self.context_loader.scan_project()
                        self.tui.print_status(f"Auto-project: {new_proj.name}")

                        # Check for logs
                        import glob

                        logs = glob.glob(
                            os.path.join(str(new_proj), "**/*.log"), recursive=True
                        )
                        if logs:
                            self.tui.print_status(
                                f"[{len(logs)} log files found — ask 'scan logs' to analyze]",
                                "dim yellow",
                            )
                    break

            # Phase 1 - CLASSIFY
            NOISE_PATTERNS = [
                r"^[^a-zA-Z0-9/~.]+$",  # pure punctuation
                r"^\w{1,8}[!\?]+$",  # single word + punctuation
            ]
            is_noise = any(re.match(p, user_input.strip()) for p in NOISE_PATTERNS)
            if is_noise:
                user_msg_raw = f"[Short social input, 1-line reply only]: {user_input}"
            else:
                user_msg_raw = user_input

            # TASK OFFLOADING: Auto-divide massive context without ceiling hits
            if len(user_msg_raw) > 3000 or user_msg_raw.count("\n") > 50:
                self.tui.print_status(
                    f"  • [color(245)]Massive input detected: {len(user_msg_raw)} chars. Offloading to task file...[/]",
                    "",
                )

                task_path = os.path.join(str(self.project_dir), ".nedster_task.md")
                try:
                    with open(task_path, "w", encoding="utf-8") as f:
                        f.write(user_msg_raw)
                    user_msg_raw = (
                        f"[SYSTEM AUTOMATION] The user pasted a massive {len(user_msg_raw)}-character task. "
                        f"To protect your context window, it was saved to `{task_path}`.\n\n"
                        f"YOUR DIRECTIVES:\n"
                        f"1. Use `read_file` to read the first part of `{task_path}`.\n"
                        f"2. Formulate a step-by-step plan.\n"
                        f"3. Execute ONLY the first 1-2 steps using your tools (e.g. `multi_edit`, `run_bash`).\n"
                        f"4. Do NOT attempt to solve the entire file in one response. Stop and ask me to continue when ready."
                    )
                except Exception as e:
                    self.tui.print_error(f"Failed to offload task: {e}")

            input_type = self._classify_input(user_msg_raw)

            # Phase 2 - CONTEXT
            context_block = ""
            if input_type in ("code", "git", "test"):
                files = self.context_loader.select_context_files(user_msg_raw)
                context_block = self.context_loader.build_context_block(files)

            # Phase 3 - PROMPT ASSEMBLY
            
            model_tier = '★★★'
            if getattr(self, 'tool_use_enabled', True) == False:
                model_tier = '★☆☆'

            system_prompt = self.build_system_prompt(tool_use_enabled=getattr(self, 'tool_use_enabled', True), model_tier=model_tier)

            # system_prompt = ANTI_MENU_INJECTION (moved to build_system_prompt)

            messages = [{"role": "system", "content": system_prompt}]
            self.hooks.fire("pre_llm_call", messages=messages, user_input=user_input)

            # Inject anchor every 8 turns as a system-role message refresh
            if self.memory.turn_count > 0 and self.memory.turn_count % 8 == 0:
                messages.append({"role": "system", "content": ANCHOR})

            messages.extend(self.memory.get_context_messages())

            # Add context and user input
            user_msg = user_msg_raw
            if context_block:
                user_msg = f"{context_block}\n\n{user_msg_raw}"

            total_chars = sum(len(m.get("content", "")) for m in messages)
            ctx_pct = min(
                100, int((total_chars / 16384) * 100)
            )  # roughly 4096 tokens * 4

            FILE_OP_PATTERNS = re.compile(
                r"\b(create|write|make|build|generate|scaffold|copy|move)\b"
                r".{0,30}\b(file|folder|directory|project|script)\b",
                re.IGNORECASE,
            )

            if FILE_OP_PATTERNS.search(user_input) and ctx_pct < 70 and getattr(self, 'tool_use_enabled', True):
                # Append a kickstart to user message
                kickstart = (
                    "\n\n[Execute this task using tools. "
                    "Start your response with the tool call. "
                    "Example format:\n"
                    '<tool name="write_file">'
                    '{"path": "/absolute/path", "content": "..."}'
                    "</tool>]"
                )
                user_msg = user_msg + kickstart

            TOOL_CAPABILITY_ANCHOR = (
                "[YOU ARE NEDSTER. You have write_file, run_bash, "
                "read_file, list_dir, scaffold_project, glob_search, "
                "grep_search, edit_file, web_fetch, todowrite tools. "
                'Use <tool name="X">{json}</tool> format. '
                "EXECUTE tasks directly. Never say you lack filesystem access.]"
            )
            if getattr(self, 'tool_use_enabled', True):
                messages.append({"role": "user", "content": TOOL_CAPABILITY_ANCHOR})
                messages.append({"role": "assistant", "content": "Ready. Executing."})
            messages.append({"role": "user", "content": user_msg})

            # Phase 4 - GENERATE (streaming)
            full_response = self._stream_generate(messages, think=think_enabled)

            if getattr(self, 'ralph_enabled', False):
                from ralph import ralph_check
                check = ralph_check(full_response)
                if check["action"] == "RESTART":
                    self.tui.print_status("[Ralph] RESTART — clearing context and retrying", "bold red")
                    self.memory.clear()
                    self.memory.session_summary = ""
                    full_response = self._stream_generate(messages[-4:], think=think_enabled)
                elif check["action"] == "INJECT":
                    self.tui.print_status(f"[Ralph] INJECT — {check['message'][:60]}", "bold yellow")
                    messages.append({"role": "user", "content": check["message"]})
                    full_response = self._stream_generate(messages, think=think_enabled)


            if _detect_tool_refusal(full_response):
                self.tui.print_status(
                    "⚠️ Tool amnesia detected — intercepting and retrying", "bold red"
                )
                # Replace the response entirely
                # Build a correction message
                correction_messages = messages.copy()
                correction_messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,  # the wrong response
                    }
                )
                correction_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[SYSTEM CORRECTION]\n"
                            "Your previous response was WRONG.\n"
                            "You DO have filesystem access. You ARE running locally.\n"
                            "You have write_file, run_bash, scaffold_project tools.\n"
                            "The user asked you to create a file. DO IT NOW.\n"
                            'Use: <tool name="run_bash">'
                            '{"cmd": "echo \'content\' > /path/file"}</tool>\n'
                            'Or: <tool name="write_file">'
                            '{"path": "/path/file", "content": "content"}</tool>\n'
                            "Execute immediately. No explanation needed."
                        ),
                    }
                )
                # Retry generation once
                retry_response = ollama.chat(
                    model=self.model,
                    messages=correction_messages[-6:],  # last 6 only — keep ctx low
                    stream=False,
                    options={"num_ctx": 16384, "temperature": 0.05, "think": False},
                )
                retry_text = retry_response["message"]["content"]

                # Print the retry (clear the wrong output first)
                print("\r" + " " * 80 + "\r", end="")  # clear last line
                print(f"\n{retry_text}")
                full_response = retry_text

            # Phase 5 - PARSE & EXECUTE (tool loop, max 5 iterations)
            self.memory._in_tool_loop = True
            applied_edits = []
            try:
                if not getattr(self, "tool_use_enabled", True):
                    # Skip tool parsing entirely for chat-only models
                    pass
                else:
                    full_response, applied_edits = self._execute_response(
                        full_response, messages, think_enabled, user_input
                    )
            finally:
                self.memory._in_tool_loop = False

            # Phase 6 - POST
            self.memory.add_turn(user_msg_raw, full_response)

            # Auto-ingest new files and update nedster_state.json
            if applied_edits:
                for edit in applied_edits:
                    path = edit.get("path")
                    if path and path.endswith(
                        (".py", ".md", ".txt", ".js", ".ts", ".go", ".rs", ".java")
                    ):
                        full_path = os.path.join(str(self.project_dir), path)
                        if os.path.exists(full_path):
                            try:
                                from ingestion import (
                                    get_text_from_file,
                                    chunk_text,
                                    embed_file_chunks,
                                )
                                import chromadb

                                client = chromadb.PersistentClient(path="./chroma_db")
                                collection = client.get_collection(name="rag_docs")
                                text = get_text_from_file(full_path)
                                if text:
                                    chunks = chunk_text(text, path)
                                    embed_file_chunks(
                                        chunks,
                                        self.retriever.embedder,
                                        collection,
                                        path,
                                    )
                                    self.tui.print_status(f"Auto-ingested {path}")
                            except Exception as e:
                                self.tui.print_warning(f"Ingestion failed: {e}")

                # Update nedster_state.json with recent changes
                if len(applied_edits) > 0:
                    try:
                        nedster_path = os.path.join(str(self.project_dir), "nedster_state.json")
                        if os.path.exists(nedster_path):
                            prompt = (
                                "Summarize these file changes concisely as 1-2 bullet points for the project log:\n"
                                + "\n".join(
                                    str(e.get("path", ""))
                                    + " - "
                                    + str(e.get("type", ""))
                                    for e in applied_edits
                                )
                            )
                            resp = ollama.generate(
                                model=self.model,
                                prompt=prompt,
                                options={
                                    "num_predict": 100,
                                    "temperature": 0.0,
                                    "think": False,
                                },
                            )
                            with open(nedster_path, "a", encoding="utf-8") as f:
                                f.write(
                                    "\n## Recent Changes\n"
                                    + resp["response"].strip()
                                    + "\n"
                                )
                    except Exception:
                        pass

            # Add to RAG memory if significant
            if len(full_response) > 100:
                try:
                    self.retriever.add_to_memory(
                        user_msg_raw, full_response, self.memory.session_id
                    )
                except Exception:
                    pass

            project_name = (
                os.path.basename(os.path.abspath(self.project_dir)) or "project"
            )

            CTX_EMERGENCY_THRESHOLD = 85
            if ctx_pct >= CTX_EMERGENCY_THRESHOLD:
                self.tui.print_status(
                    f"⚠️ Context {ctx_pct}% — auto-compacting to preserve tool access",
                    "bold yellow",
                )
                self.memory._compress_session()

                TOOL_EMERGENCY_REMINDER = """
[CRITICAL SYSTEM REMINDER — YOU ARE NEDSTER]
You are a LOCAL CODING AGENT with FULL filesystem access.
You HAVE these working tools RIGHT NOW:

write_file(path, content)  → CREATE any file on disk
run_bash(cmd)              → EXECUTE any shell command
read_file(path)            → READ any file
list_dir(path)             → LIST directory contents
scaffold_project(path, files) → CREATE entire project
glob_search(pattern)       → SEARCH files by pattern
grep_search(pattern)       → SEARCH content in files
edit_file(path, old, new)  → EDIT existing file
web_fetch(url)             → FETCH url
todowrite(todos)           → TRACK tasks

To create /home/mnm/Downloads/hello.txt:
<tool name="run_bash">{"cmd": "echo 'Hello!' > /home/mnm/Downloads/hello.txt"}</tool>

DO NOT say "I cannot create files."
DO NOT say "I am an AI without filesystem access."
EXECUTE using the tool format above. RIGHT NOW.
"""
                messages.insert(
                    -1, {"role": "user", "content": TOOL_EMERGENCY_REMINDER}
                )
                messages.insert(
                    -1,
                    {
                        "role": "assistant",
                        "content": "Understood. I will use my tools to execute this task directly.",
                    },
                )
                self.tui.print_status(
                    "⚠️ HIGH CONTEXT — suggest /clear after this response", "bold yellow"
                )

            self.tui.print_status_bar(
                project_name,
                self.model,
                getattr(self, "model_size_gb", 0.0),
                ctx_pct,
                self.tool_stats["calls"],
                self.tool_stats["edits"],
                think_enabled,
            )
            self._last_ctx_pct = ctx_pct

            if ctx_pct >= 95:
                print("\n⚠️ Context at 95% — auto-clearing to restore tool access")
                self.memory.clear()
                self.memory.session_summary = ""
                _seen_this_iteration = set()
                print(
                    "[Memory cleared automatically. Use /compact to summarize first next time.]"
                )

        finally:
            pass

    def _classify_input(self, user_input: str) -> str:
        """
        Classify input type:
        - "code task": mentions file extension, edit, fix, write, implement
        - "git task": commit, branch, status, diff
        - "test task": test, run, check, failing
        - "question": general question
        """
        lower = user_input.lower()

        # Code indicators
        code_patterns = [
            ".py",
            ".js",
            ".ts",
            ".go",
            ".rs",
            ".java",
            "fix",
            "edit",
            "write",
            "implement",
            "add",
            "remove",
            "function",
            "class",
            "import",
            "def ",
            "return",
        ]
        if any(p in lower for p in code_patterns):
            return "code"

        # Git indicators
        git_patterns = [
            "commit",
            "branch",
            "status",
            "diff",
            "stash",
            "git ",
            "push",
            "pull",
            "merge",
            "rebase",
        ]
        if any(p in lower for p in git_patterns):
            return "git"

        # Test indicators
        test_patterns = [
            "test",
            "pytest",
            "npm test",
            "failing",
            "passed",
            "run the tests",
            "check syntax",
        ]
        if any(p in lower for p in test_patterns):
            return "test"

        return "question"



    def _detect_loop_enhanced(self, accumulated: str) -> bool:
        _LOOP_CHAR_THRESH = 30
        _LOOP_PHRASE_THRESH = 3
        if len(accumulated) < 20: return False

        tail = accumulated[-_LOOP_CHAR_THRESH:]
        if len(set(tail[::2])) <= 2: return True

        chunks = [accumulated[i:i+40].strip() for i in range(0, len(accumulated)-40, 40)]
        if len(chunks) >= _LOOP_PHRASE_THRESH:
            from collections import Counter
            counts = Counter(chunks)
            most_common_count = counts.most_common(1)[0][1]
            if most_common_count >= _LOOP_PHRASE_THRESH: return True

        words = accumulated.split()
        if len(words) > 15:
            recent_10 = words[-10:]
            if len(set(recent_10)) <= 3: return True

        return False

    def _strip_weak_model_artifacts(self, text: str) -> str:
        """
        Strip artifacts leaked by weak/chat-only models that hallucinate
        system-prompt fragments into their output.

        FIX: The original had orphaned code (lines checking `accumulated` and
        `_REPEAT_THRESHOLD`) placed AFTER `return text.strip()`. That dead code
        referenced an undefined variable and would have raised NameError if
        somehow reached. Removed entirely.
        """
        import re
        text = re.sub(r'<tool\s+name="[^"]*">.*?</tool>', '', text, flags=re.DOTALL)
        text = re.sub(r'\[YOU ARE NEDSTER\..*?\]', '', text, flags=re.DOTALL)
        text = re.sub(r'YOU ARE NEDSTER[.,].*?(?=\n|$)', '', text, flags=re.MULTILINE)
        text = re.sub(r'={3,}\s*FILE:.*?={3,}', '', text, flags=re.DOTALL)
        text = re.sub(r'\*\*Final response:\*\*\s*', '', text)
        text = re.sub(r'\*\*Final reply:\*\*\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        # Strip leaked system-prompt fragments from chat-only models
        text = re.sub(r'DIRECTIVE (?:ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT).*?(?=\n\n|$)', '', text, flags=re.DOTALL)
        text = re.sub(r'\[NON-NEGOTIABLE EXECUTION RULES\].*?(?=\n\n|$)', '', text, flags=re.DOTALL)
        return text.strip()
    def _stream_generate(self, messages: List[dict], think: bool = False) -> str:
        """
        Generate response with streaming output.
        Returns full response text.
        """
        full_response = ""

        try:
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "num_ctx": 16384,
                    "num_predict": 2048,
                    "temperature": 0.10,
                    "think": think,
                },
            )

            in_think = False
            printed_idx = 0

            _token_count = 0
            _empty_chunks = 0
            _last_chunk = ""
            _repeat_count = 0
            REPEAT_THRESHOLD = 5
            MAX_RESPONSE_TOKENS = 2048

            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    _token_count += 1

                    if _token_count > MAX_RESPONSE_TOKENS:
                        self.tui.print_warning(
                            "Response ceiling hit — stopping generation."
                        )
                        try:
                            stream.close()
                        except:
                            pass
                        break

                    raw = content
                    if raw == _last_chunk:
                        _repeat_count += 1
                        if _repeat_count >= REPEAT_THRESHOLD:
                            self.tui.print_warning(
                                "Repetition loop detected — stopping."
                            )
                            try:
                                stream.close()
                            except:
                                pass
                            break
                    else:
                        _repeat_count = 0
                    _last_chunk = raw

                    stripped_raw = raw.strip().replace("</think>", "").strip()
                    if not stripped_raw:
                        _empty_chunks += 1
                        if _empty_chunks > 10:
                            self.tui.print_warning("Think loop detected — breaking.")
                            try:
                                stream.close()
                            except:
                                pass
                            break
                    else:
                        _empty_chunks = 0

                    full_response += content

                    # Strip orphaned </think> that arrive without matching <think>
                    if (
                        not in_think
                        and "</think>" in full_response
                        and "<think>" not in full_response
                    ):
                        full_response = full_response.replace("</think>", "")

                    is_thinking = (
                        "<think>" in full_response and "</think>" not in full_response
                    )
                    if is_thinking:
                        if not in_think:
                            self.tui.console.print(
                                "  • Thinking...",
                                style=self.tui.COLORS["thinking"],
                                end="\r",
                            )
                            in_think = True

                        continue

                    if in_think and "</think>" in full_response:
                        self.tui.console.print(" " * 20, end="\r")
                        in_think = False

                    content_clean = re.sub(
                        r"<think>.*?</think>\n*", "", full_response, flags=re.DOTALL
                    )
                    new_clean = content_clean[printed_idx:]
                    if new_clean:
                        self.tui.print_response_stream(new_clean)
                        printed_idx = len(content_clean)

        except Exception as e:
            error_msg = f"[Generation error: {e}]"
            self.tui.print_error(error_msg)
            return error_msg

        print()  # Newline after streaming

        CORRUPTION_TELLS = [
            "as an AI",
            "I'm an AI",
            "I cannot access",
            "without filesystem",
            "I don't have the ability",
            "I'm just",
            "I'm unable to",
            "You would need to",
        ]
        if any(t.lower() in full_response.lower() for t in CORRUPTION_TELLS):
            print("\n[Nedster] ⚠️ Context corruption detected.")
            print("[Nedster] Run /clear then retry your request.")

        size_gb = self._get_model_size(self.model)
        if size_gb > 0:
            self.model_size_gb = size_gb

        full_response = self._strip_weak_model_artifacts(full_response)
        # [fixer] summary gate
        if self._needs_summary(full_response, tool_results_str if 'tool_results_str' in dir() else ''):
            full_response = full_response.rstrip() + '\n' + self._generate_summary(user_input if 'user_input' in dir() else '', tool_results_str if 'tool_results_str' in dir() else '')
        return full_response


    def _extract_prose(self, text: str) -> str:
        """Return only the non-tool-call text from a response."""
        import re
        cleaned = re.sub(r'<tool\s+name="[^"]*">.*?</tool>', '', text, flags=re.DOTALL)
        cleaned = re.sub(r'<!--.*?-->', '', cleaned, flags=re.DOTALL)
        return cleaned.strip()

    def _needs_summary(self, response: str, tool_results: str) -> bool:
        """
        Returns True when the model ran tools but gave no prose reply.
        This is the 'silent scaffolding' bug.
        """
        prose = self._extract_prose(response)
        has_tool_calls = '<tool name=' in response
        meaningful_prose = len([c for c in prose if c.isalpha()]) > 15
        return has_tool_calls and not meaningful_prose

    def _generate_summary(self, user_input: str, tool_results: str) -> str:
        """
        Generate a short completion summary when the agent
        ran tools but produced no natural language response.
        Uses a tight 200-token budget — just enough for a summary.
        """
        import ollama
        import re
        files_mentioned = re.findall(r'(?:Written|Created|Read):\s*([^\s\(]+)', tool_results)
        files_str = (", ".join(files_mentioned[:5]) if files_mentioned else "")

        summary_prompt = (
            f"Task was: {user_input[:100]}\n\n"
            f"Tool results summary:\n{tool_results[:600]}\n\n"
            f"In 1-3 lines, state what was done. "
            f"{'Files: ' + files_str + '. ' if files_str else ''}"
            f"Be direct. No bullet lists. No questions. "
            f"No 'What would you like to do next?'"
        )

        try:
            resp = ollama.generate(
                model=self.model,
                prompt=summary_prompt,
                options={
                    "num_ctx": 16384,
                    "num_predict": 120,
                    "temperature": 0.1,
                    "think": False,
                }
            )
            summary = resp.get("response", "").strip()
            summary = self._extract_prose(summary)
            return summary if len(summary) > 5 else "Done."
        except Exception:
            return "Done."

    def _assess_response_quality(self, response: str, tool_results: str) -> str:
        """
        Detect and fix low-quality responses.
        Returns improved response or original if quality is OK.
        """
        import re
        prose = self._extract_prose(response)

        NARRATION_PATTERNS = [
            r"I(?:\\'m| am) (?:now |going to |about to )?(?:reading|checking|looking|scanning|running)",
            r"Let me (?:read|check|look|scan|run|examine)",
            r"I(?:\\'ll| will) (?:read|check|look|scan|run)",
            r"\n\s*\d+\.\s+\*\*[A-Z][^*]+\*\*\s*[-—]\s*",
            r"What (?:specific|would you like|aspect|functionality)",
            r"What would you like (?:me to|to)",
            r"I can help you with:",
            r"Here's (?:what I can|a comprehensive overview)",
        ]
        COMPLETION_PATTERNS = [
            r"(?:scaffold|project) (?:created|initialized|ready)",
            r"Written:\s+\S+",
            r"\d+ files? (?:created|written)",
            r"cargo init",
            r"\[exit 0\]",
        ]

        is_narrating = any(re.search(p, prose, re.IGNORECASE) for p in NARRATION_PATTERNS)
        task_done = any(re.search(p, tool_results, re.IGNORECASE) for p in COMPLETION_PATTERNS)

        if is_narrating and task_done:
            return self._generate_summary(self.memory.short_term[-1]["content"] if self.memory.short_term else "", tool_results)

        return response

    def _execute_response(
        self, response: str, messages: List[dict], think: bool, user_input: str
    ) -> tuple[str, list]:
        """
        Parse and execute tool calls and edit blocks.
        Max 10 iterations via IterationBudget.

        FIX: The original code created a ThreadPoolExecutor but never submitted
        futures to it — `futures` was always an empty dict, so the
        `as_completed(futures)` loop did nothing. Additionally, tools NOT in
        SAFE_PARALLEL fell into the `else` branch which only printed a warning.
        Result: no tool ever produced output to tool_msg_accumulator.

        FIXED APPROACH:
        - All tools execute via self.executor.execute() immediately.
        - SAFE_PARALLEL tools still run synchronously for simplicity
          (parallelism was broken anyway; real parallel support requires
          submitting to futures dict first).
        - Every result is accumulated into tool_msg_accumulator.
        """
        budget = IterationBudget(max_iters=10, max_chars=12000)
        final_response = response
        seen_tool_calls = set()
        applied_edits = []
        WATCHDOG.start()

        try:
            tool_loops = 0
            while budget.remaining > 0:
                iteration = budget._iters + 1
                tool_loops += 1
                self.tool_stats["loops"] += 1
                _seen_this_iteration = set()

                # Parse edit blocks
                edits = self.editor.parse_edit_blocks(response)

                # Parse tool calls
                tool_calls = parse_tool_calls(response)

                if not edits and not tool_calls:
                    break

                # Execute edits
                for edit in edits:
                    result = self.editor.apply_edit(edit, auto=self.auto)
                    self.tui.print_status(f"Edit: {result}")
                    if "Edited" in result or "Created" in result or "Overwritten" in result:
                        self.tool_stats["edits"] += 1
                        applied_edits.append(edit)

                    # Auto-check syntax for Python files
                    if edit.get("path", "").endswith(".py"):
                        syntax_result = self._check_file_syntax(str(edit.get("path", "")))
                        if syntax_result != "OK":
                            self.tui.print_warning(f"Syntax issue: {syntax_result}")

                tool_msg_accumulator = ""

                # Execute tool calls
                import json
                from tools import TOOL_REGISTRY, TOOL_NAME_ALIASES, normalize_tool_args

                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "")
                    args = normalize_tool_args(tool_name, tool_call.get("args", {}))

                    raw_name = tool_name
                    # Normalize tool name
                    t_name = raw_name.strip().lower().replace("-", "_")
                    if t_name not in TOOL_REGISTRY:
                        aliased = TOOL_NAME_ALIASES.get(t_name)
                        if aliased is None and t_name in TOOL_NAME_ALIASES:
                            continue  # explicitly discarded
                        if aliased and aliased in TOOL_REGISTRY:
                            t_name = aliased
                        else:
                            self.tui.print_status(
                                f"[BLOCKED] '{raw_name}' not a valid tool", "bold red"
                            )
                            tool_msg_accumulator += (
                                f"[ERROR: '{raw_name}' unknown. Use write_file to create files.]\n"
                            )
                            continue

                    tool_name = t_name  # use normalized name

                    call_hash = f"{tool_name}:{json.dumps(args, sort_keys=True)}"

                    NEVER_DEDUP = {"list_dir", "git_status", "run_bash", "read_file"}
                    if tool_name not in NEVER_DEDUP:
                        if call_hash in _seen_this_iteration:
                            self.tui.print_status(
                                f"[SKIP] Identical call in same batch: {tool_name}", "dim"
                            )
                            continue
                        _seen_this_iteration.add(call_hash)

                    self.tui.print_tool_call(name=tool_name, args=args)

                    # FIX: Execute ALL tools through executor and collect results.
                    # Previously SAFE_PARALLEL tools ran but result was discarded;
                    # non-SAFE_PARALLEL tools only printed a warning and never ran.
                    if tool_name in TOOL_REGISTRY:
                        # Inject cwd for git tools
                        if "cwd" not in args and tool_name.startswith("git_"):
                            args["cwd"] = self.project_dir

                        result = self.executor.execute(tool_name, args, self.tui)
                        self.tui.print_tool_result(tool_name, result, verbose=self.verbose)
                        self.tool_stats["calls"] += 1
                        tool_msg_accumulator += f"[Tool result: {tool_name}]\n{result}\n\n"
                        WATCHDOG.ping()
                    else:
                        # Should not happen after alias resolution above, but guard anyway
                        self.tui.print_warning(f"Unknown tool after normalization: {tool_name}")
                        tool_msg_accumulator += f"[ERROR: '{tool_name}' not in registry]\n"

                # If nothing ran at all, break
                if not tool_msg_accumulator and not edits:
                    break

                # Regenerate response with tool results
                if tool_msg_accumulator:
                    preview = (
                        tool_msg_accumulator[:70]
                        .replace("\n", " ")
                        .replace("[Tool result: ", "")
                        .replace("]", "")
                    )
                    self.tui.print_status(
                        f"  • [{self.tui.COLORS['tool']}]Result: {preview}...[/]", ""
                    )

                    if not budget.consume(messages):
                        tool_msg_accumulator += budget.inject_limit_message()
                        break

                    tool_msg_accumulator += _build_verification_injection(tool_msg_accumulator)
                    tool_msg_accumulator += "\nContinue."

                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": tool_msg_accumulator})
                    response = self._stream_generate(messages, think=think)
                    final_response += "\n\n" + response
                else:
                    break

                # After edits, offer to run tests
                if edits and iteration == 1:
                    test_runner = self._detect_test_runner()
                    if test_runner != "unknown":
                        self.tui.print_status(f"Test runner detected: {test_runner}")
                        if not self.auto:
                            try:
                                resp = input("Run tests? [y/N] ").strip().lower()
                                if resp in ("y", "yes"):
                                    test_result = self._run_tests()
                                    self.tui.print_status(
                                        f"Tests: {test_result[:200]}..."
                                    )
                                    self.tool_stats["tests"] += 1
                            except (EOFError, KeyboardInterrupt):
                                pass

        finally:
            WATCHDOG.stop()

        verification_warning = _verify_task_completion(user_input, final_response)
        if verification_warning:
            print(verification_warning)
            messages.append({"role": "assistant", "content": final_response})
            messages.append(
                {
                    "role": "user",
                    "content": verification_warning
                    + "\nDo NOT confirm success. Fix the incomplete task now.",
                }
            )
            final_response += "\n\n" + self._stream_generate(messages, think=think)

        return final_response, applied_edits
    def _check_file_syntax(self, path: str) -> str:
        """Check syntax of a file."""
        try:
            if path.endswith(".py"):
                with open(path, "r", encoding="utf-8") as f:
                    code = f.read()
                from code_tools import check_syntax

                return check_syntax(code, "python")
        except Exception as e:
            return f"Error: {e}"
        return "OK"

    def _detect_test_runner(self) -> str:
        """Detect test runner."""
        try:
            from code_tools import detect_test_runner

            return detect_test_runner(self.project_dir)
        except Exception:
            return "unknown"

    def _run_tests(self) -> str:
        """Run tests."""
        try:
            from code_tools import run_tests

            return run_tests(self.project_dir)
        except Exception as e:
            return f"Error: {e}"

    def save_session(self):
        """Force save session to milestones and journal."""
        if not self.memory.short_term:
            return "No messages to save."

        try:
            from journal import score_session

            prompt = (
                "Extract 2-3 single-line bullet points summarizing key technical facts, decisions, "
                "or paths used in this session. Return ONLY the bullet points, no preamble:\n"
                f"{self.memory.get_last_n_turns_text(n=5)}"
            )
            resp = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_ctx": 16384,
                    "num_predict": 150,
                    "temperature": 0.0,
                    "think": False,
                },
            )
            milestones = resp["response"].strip()

            from datetime import datetime

            os.makedirs(os.path.expanduser("~/.aria"), exist_ok=True)

            date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            topic = "Project update"
            if len(self.memory.short_term) > 0:
                first_msg = self.memory.short_term[0]["content"]
                topic = first_msg[:50] + ("..." if len(first_msg) > 50 else "")
                topic = topic.replace("\n", " ").strip()

            # OPENCLAW SESSIONS INTEGRATION
            session_meta = {
                "id": self.memory.session_id,
                "date": date_str,
                "topic": topic,
                "summary": milestones,
                "quality": score_session(
                    self.memory.short_term, tasks_completed=self.tool_stats["edits"]
                ),
            }
            import json

            idx_path = os.path.expanduser("~/.aria/sessions_index.json")
            sessions = []
            if os.path.exists(idx_path):
                try:
                    with open(idx_path, "r") as f:
                        sessions = json.load(f)
                except Exception:
                    pass
            sessions.append(session_meta)
            with open(idx_path, "w") as f:
                json.dump(sessions, f, indent=2)

            with open(os.path.expanduser("~/.aria/milestones.md"), "a") as f:
                f.write(
                    f"\n## Session {self.memory.session_id}\n"
                    f"Date: {date_str}  |  Quality: {session_meta['quality']}/5\n"
                    f"Topic: {topic}\n"
                    f"{milestones}\n"
                )

            return f"Session {self.memory.session_id} saved to milestones."
        except Exception as e:
            return f"Error saving session: {e}"

    def plan_and_execute(self, task: str):
        """
        For complex multi-file tasks:
        1. Ask LLM to produce a numbered plan
        2. Show plan to user, confirm (unless auto mode)
        3. Execute each step in sequence
        4. After all steps: run_tests() + run_linter()
        5. git_status() to show what changed
        6. Offer git_commit() with auto-generated message
        """
        # Generate plan
        plan_prompt = f"""Break this task into numbered steps (3-5 max). Output ONLY the steps:
{task}

Format:
Step 1: [action] in [file/location]
Step 2: [action] in [file/location]
...
"""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=plan_prompt,
                options={
                    "num_ctx": 16384,
                    "num_predict": 200,
                    "temperature": 0.0,
                    "think": False,
                },
            )
            plan_text = response["response"].strip()

            # Parse steps
            steps = []
            for line in plan_text.split("\n"):
                line = line.strip()
                if line.startswith("Step") or (line[0].isdigit() and "." in line[:3]):
                    steps.append(line)

            if not steps:
                steps = [plan_text]

            # Show plan
            self.tui.print_status("Plan:")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step}")

            # Confirm unless auto mode
            if not self.auto:
                try:
                    resp = input("\nExecute plan? [Y/n] ").strip().lower()
                    if resp not in ("", "y", "yes"):
                        self.tui.print_status("Plan cancelled")
                        return
                except (EOFError, KeyboardInterrupt):
                    self.tui.print_status("Plan cancelled")
                    return

            # Execute steps
            for i, step in enumerate(steps, 1):
                self.tui.print_status(f"Step {i}/{len(steps)}: {step}")
                self.generate(step)

            # Post-execution: run tests and linter
            self.tui.print_status("Running tests...")
            test_result = self._run_tests()
            self.tui.print_status(f"Tests: {test_result[:100]}...")

            self.tui.print_status("Running linter...")
            try:
                from code_tools import run_linter

                lint_result = run_linter(self.project_dir)
                self.tui.print_status(f"Lint: {lint_result[:100]}...")
            except Exception:
                pass

            # Show git status
            self.tui.print_status("Git status:")
            try:
                from git_tools import git_status

                status = git_status(self.project_dir)
                print(status[:500])
            except Exception:
                pass

            # Offer commit
            if not self.auto:
                try:
                    resp = input("\nCommit changes? [y/N] ").strip().lower()
                    if resp in ("y", "yes"):
                        msg = input("Commit message (or empty for auto): ").strip()
                        try:
                            from git_tools import git_commit

                            result = git_commit(self.project_dir, msg)
                            self.tui.print_success(result)
                        except Exception as e:
                            self.tui.print_error(f"Commit failed: {e}")
                except (EOFError, KeyboardInterrupt):
                    pass

        except Exception as e:
            self.tui.print_error(f"Plan execution failed: {e}")



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
        "<tool name=\"X\">",
    ]
    for bp in BAD_PREFIXES:
        if accumulated.count(bp) >= 2:
            return True

    return False

# ─────────────────────────────────────────────────────────────────────────────



# ── nedster_fixer: output strip ───────────────────────────────────────────────

def _strip_model_artifacts(text: str) -> str:
    """Remove hallucinated tool XML and identity anchors from model output."""
    import re as _re
    # Raw tool call XML echoed by weak models
    text = _re.sub(r'<tool\s+name="[^"]*">.*?</tool>', '', text, flags=_re.DOTALL)
    # [YOU ARE NEDSTER. ...] echoed from system prompt
    text = _re.sub(r'\[YOU ARE NEDSTER\..*?\]', '', text, flags=_re.DOTALL)
    text = _re.sub(r'YOU ARE NEDSTER[.,][^\n]*', '', text)
    # === FILE: ... === echoed format markers
    text = _re.sub(r'={3,}\s*FILE:.*?={3,}', '', text, flags=_re.DOTALL)
    # **Final response:** / **Final reply:**
    text = _re.sub(r'\*\*Final (?:response|reply):\*\*\s*', '', text)
    # Trailing open code fence
    text = _re.sub(r'```\s*$', '', text)
    return text.strip()

# ─────────────────────────────────────────────────────────────────────────────



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
