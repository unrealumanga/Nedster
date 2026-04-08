import os

"""Nedster Agent - Core agentic loop extending RAG pipeline"""

import re
import ollama
from datetime import datetime
from typing import Optional, Dict, List

from context_loader import ContextLoader
from editor import FileEditor
from retriever import Retriever
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


class NedsterAgent:
    """
    NedsterAgent - Local coding agent with RAG, context awareness, and tool use.
    Extends the Aria RAG pipeline with project context and code editing.
    """

    NEDSTER_SYSTEM_PROMPT = """DIRECTIVE ZERO — LANGUAGE:
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

DIRECTIVE SIX — NEDSTER.md:
On task completion, extract any project facts:
  architecture decisions, dependencies added, patterns used.
Append to NEDSTER.md silently. Say: "Project memory updated."

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
        self.project_dir = project_dir
        self.auto = auto
        self.think = think
        self.verbose = False
        self.model = "aria-qwen"

        self.context_loader = ContextLoader(project_dir)
        self.editor = FileEditor(project_dir)
        self.retriever = Retriever()
        self.memory = MemoryManager(self.model)
        self.tui = NedsterTUI()

        self.tool_stats = {"calls": 0, "loops": 0, "edits": 0, "tests": 0}
        self.pending_plan: Optional[str] = None
        self.plan_steps: List[str] = []
        self.current_step = 0

        self._boot_project()

    def _boot_project(self):
        """
        1. context_loader.scan_project()
        2. Read NEDSTER.md - inject into system prompt
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

        # Read NEDSTER.md
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
            self.tui.print_status("Project memory: NEDSTER.md loaded")

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
                        from tools import SESSION

                        SESSION.set_project(str(new_proj))
                        self.project_dir = str(new_proj)
                        self.context_loader.project_root = new_proj
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
            system_prompt = self.NEDSTER_SYSTEM_PROMPT

            # Load Global Skills (Self-Taught)
            skills_dir = os.path.expanduser("~/.agents/skills/")
            global_skills = ""
            if os.path.exists(skills_dir):
                import glob
                # Prioritize user-created skills over built-in agent skills to prevent overflow
                user_skills = glob.glob(os.path.join(skills_dir, "user/**/*.md"), recursive=True)
                other_skills = glob.glob(os.path.join(skills_dir, "*.md"), recursive=False)
                
                skill_files = user_skills + other_skills
                
                for sf in skill_files:
                    if len(global_skills) > 4000:  # STRICT CAP: Prevent context overflow
                        break
                    try:
                        with open(sf, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                skill_name = os.path.basename(os.path.dirname(sf))
                                if skill_name == "user" or skill_name == "skills":
                                    skill_name = os.path.basename(sf).replace(".md", "")
                                # Take max 800 chars per skill to fit more
                                global_skills += f"\n--- SKILL: {skill_name} ---\n{content[:800]}\n"
                    except Exception:
                        pass
                        
            if global_skills:
                # Absolute hard cap on the injected string length
                if len(global_skills) > 5000:
                    global_skills = global_skills[:5000] + "\n...[Additional skills truncated]"
                system_prompt += f"\n\n## Global Skills (Learned Capabilities):\n{global_skills}"

            # Add NEDSTER.md content
            nedster_md = self.context_loader.read_nedster_md()
            if nedster_md:
                system_prompt += f"\n\n## Project Memory:\n{nedster_md[:2000]}"

            # Add tool inventory
            tool_list = "\n".join(f"  - {name}" for name in TOOL_REGISTRY.keys())
            system_prompt += f"\n\n## Available Tools:\n{tool_list}"

            POWER_TOOLS = """
POWER TOOLS — use these before brute-force exploration:

  context_inject(mode="project", path=X)
    → Call FIRST when switching to any project directory.
      Gives you full architecture in one call.
      Replaces 10+ list_dir + read_file calls.

  context_inject(mode="task", path=X, query="your task")
    → Call before starting any coding task.
      Finds the most relevant files automatically.

  context_inject(mode="bot", path=X)
    → Call when H2 mentions a trading bot.
      Gives structure + log summary in one call.

  codebase_map(path=X)
    → Full project tree + language stats + entry points.

  code_xray(path="file.py", focus="security")
    → Deep analysis without reading raw file.
      Use instead of read_file for files > 100 lines.

  log_analyzer(path=X, mode="pnl")
    → Extract PNL/win-rate from trading bot logs.
      Use instead of read_file on .log files.

  market_intel(symbol="BTC", depth=True)
    → Real-time price, funding, OI, orderbook.
      No API key needed. Replaces get_crypto_price.

  multi_edit(edits=[...])
    → Apply multiple file changes atomically.
      Replaces multiple write_file calls with rollback.

  process_watch(action="status")
    → See all running bots + VRAM consumers at once.

  bot_runner(action="start", bot_path=X)
    → Launch any bot in tmux. Survives terminal close.

  model_bench(model="qwen3.5:9b")
    → Measure tok/sec before recommending a model switch.

  secret_scan(path=X)
    → Scan for exposed credentials before any commit.
"""
            system_prompt += "\n" + POWER_TOOLS

            ANCHOR = """
[ACTIVE DIRECTIVES REMINDER]
- English only. Mirror H2 message length.
- No emojis except ⚠️. No numbered menus. No "Great question!"
- Short input = action command. Execute, don't explain.
- Emit <edit> or <tool> blocks for ALL file/code changes.
- You are Nedster. H2 is your user.
"""
            system_prompt += "\n" + ANCHOR

            # Build messages
            messages = [{"role": "system", "content": system_prompt}]

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

            if FILE_OP_PATTERNS.search(user_input) and ctx_pct < 70:
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
            messages.append({"role": "user", "content": TOOL_CAPABILITY_ANCHOR})
            messages.append({"role": "assistant", "content": "Ready. Executing."})
            messages.append({"role": "user", "content": user_msg})

            # Phase 4 - GENERATE (streaming)
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
                    options={"num_ctx": 2048, "temperature": 0.05, "think": False},
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

            # Auto-ingest new files and update NEDSTER.md
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

                # Update NEDSTER.md with recent changes
                if len(applied_edits) > 0:
                    try:
                        nedster_path = os.path.join(str(self.project_dir), "NEDSTER.md")
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
                    "num_ctx": 4096,
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

        return full_response

    def _execute_response(
        self, response: str, messages: List[dict], think: bool, user_input: str
    ) -> tuple[str, list]:
        """
        Parse and execute tool calls and edit blocks.
        Max 3 iterations.
        """
        max_iterations = 15
        iteration = 0
        final_response = response
        seen_tool_calls = set()
        applied_edits = []

        WATCHDOG.start()
        try:
            tool_loops = 0
            while iteration < max_iterations:
                iteration += 1
                tool_loops += 1
                self.tool_stats["loops"] += 1
                _seen_this_iteration = set()

                # Parse edit blocks
                edits = self.editor.parse_edit_blocks(response)

                # Parse tool calls
                tool_calls = parse_tool_calls(response)

                if not edits and not tool_calls:
                    # No actions to execute
                    break

                # Execute edits
                for edit in edits:
                    result = self.editor.apply_edit(edit, auto=self.auto)
                    self.tui.print_status(f"Edit: {result}")
                    if (
                        "Edited" in result
                        or "Created" in result
                        or "Overwritten" in result
                    ):
                        self.tool_stats["edits"] += 1
                        applied_edits.append(edit)

                    # Auto-check syntax for Python files
                    if edit.get("path", "").endswith(".py"):
                        syntax_result = self._check_file_syntax(edit.get("path"))
                        if syntax_result != "OK":
                            self.tui.print_warning(f"Syntax issue: {syntax_result}")

                tool_msg_accumulator = ""
                # Execute tool calls
                import json
                import concurrent.futures

                SAFE_PARALLEL = {
                    "read_file",
                    "list_dir",
                    "search_code",
                    "get_crypto_price",
                    "duckduckgo_search",
                    "smart_search",
                    "tavily_search",
                    "glob_search",
                    "grep_search",
                    "web_fetch",
                }

                # Process sequentially but track safe calls for parallel execution
                futures = {}
                ex = None

                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "")
                    from tools import normalize_tool_args

                    args = normalize_tool_args(tool_name, tool_call.get("args", {}))

                    from tools import TOOL_REGISTRY, TOOL_NAME_ALIASES

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
                            tool_msg_accumulator += f"[ERROR: '{raw_name}' unknown. Use write_file to create files.]\n"
                            continue

                    tool_name = t_name  # use normalized name

                    call_hash = f"{tool_name}:{json.dumps(args, sort_keys=True)}"

                    NEVER_DEDUP = {"list_dir", "git_status", "run_bash", "read_file"}
                    if tool_name not in NEVER_DEDUP:
                        if call_hash in _seen_this_iteration:
                            self.tui.print_status(
                                f"[SKIP] Identical call in same batch: {tool_name}",
                                "dim",
                            )
                            continue
                        _seen_this_iteration.add(call_hash)

                    self.tui.print_tool_call(name=tool_name, args=args)

                    if tool_name in TOOL_REGISTRY:
                        # Inject cwd if not provided
                        if "cwd" not in args and tool_name.startswith("git_"):
                            args["cwd"] = self.project_dir

                        if tool_name in SAFE_PARALLEL:
                            if ex is None:
                                ex = concurrent.futures.ThreadPoolExecutor(
                                    max_workers=4
                                )

                            func = TOOL_REGISTRY[tool_name]

                            def wrapper(f=func, a=args):
                                try:
                                    return f(**a)
                                except TypeError:
                                    return f(a)

                            future = ex.submit(wrapper)
                            futures[future] = tool_name
                        else:
                            # Sequential execution for non-safe tools
                            try:
                                func = TOOL_REGISTRY[tool_name]
                                try:
                                    result = func(**args)
                                except TypeError:
                                    result = func(args)

                                self.tui.print_tool_result(
                                    tool_name, result, verbose=self.verbose
                                )

                                self.tool_stats["calls"] += 1
                                tool_msg_accumulator += (
                                    f"[Tool result: {tool_name}]\n{result}\n\n"
                                )

                                VERIFY_AFTER = {
                                    "write_file",
                                    "write",
                                    "create_file",
                                    "create file",
                                    "_create_file",
                                    "edit_file",
                                    "multi_edit",
                                }
                                if (
                                    tool_name in VERIFY_AFTER
                                    or raw_name in VERIFY_AFTER
                                ):
                                    import os
                                    from tools import SESSION

                                    written_path = args.get("path", "")
                                    if written_path:
                                        written_path = os.path.expanduser(written_path)
                                        if not os.path.isabs(written_path):
                                            written_path = os.path.join(
                                                SESSION.active_project_dir, written_path
                                            )
                                        if os.path.exists(written_path):
                                            size = os.path.getsize(written_path)
                                            tool_msg_accumulator += f"\n[AUTO-VERIFY] ✓ {written_path} exists ({size} bytes)\n"
                                        else:
                                            tool_msg_accumulator += f"\n[AUTO-VERIFY] ✗ {written_path} NOT FOUND after write — write failed silently\n"

                            except Exception as e:
                                self.tui.print_error(f"Tool {tool_name} failed: {e}")
                    else:
                        self.tui.print_warning(f"Unknown tool: {tool_name}")

                # Collect parallel results
                if ex is not None:
                    for future in concurrent.futures.as_completed(futures):
                        tool_name = futures[future]
                        try:
                            result = future.result()
                            self.tui.print_tool_result(
                                tool_name, result, verbose=self.verbose
                            )
                            self.tool_stats["calls"] += 1
                            tool_msg_accumulator += (
                                f"[Tool result: {tool_name}]\n{result}\n\n"
                            )
                        except Exception as e:
                            self.tui.print_error(f"Tool {tool_name} failed: {e}")
                    ex.shutdown()

                # After processing edits and tool calls, if we didn't update response, we must break
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

                    total_chars = sum(len(m.get("content", "")) for m in messages)

                    if iteration >= max_iterations:
                        tool_msg_accumulator += "\n[TOOL LIMIT REACHED. Stop calling tools. Summarize what you found and answer directly.]"
                    elif total_chars > 12000:
                        tool_msg_accumulator += "\n[CONTEXT LIMIT WARNING. Stop calling tools. Summarize and answer directly.]"
                    else:
                        tool_msg_accumulator += _build_verification_injection(
                            tool_msg_accumulator
                        )
                        tool_msg_accumulator += "\nContinue."

                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": tool_msg_accumulator})
                    response = self._stream_generate(messages, think=think)
                    final_response += "\n\n" + response

                    if iteration >= max_iterations or total_chars > 12000:
                        break
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
            from journal import score_session, log_session

            prompt = (
                "Extract 2-3 single-line bullet points summarizing key technical facts, decisions, "
                "or paths used in this session. Return ONLY the bullet points, no preamble:\n"
                f"{self.memory.get_last_n_turns_text(n=5)}"
            )
            resp = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_ctx": 2048,
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
                    "num_ctx": 1024,
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
