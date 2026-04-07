════════════════════════════════════════════════════════════════
PROJECT: NEDSTER — Local Claude Code Clone
Backend: Ollama + aria-qwen:latest (local Qwen3.5:9b with RAG)
Foundation: Extend the existing Aria RAG stack (combined3.txt)
════════════════════════════════════════════════════════════════

You are building "Nedster" — a Claude Code clone that runs entirely
locally using Ollama + aria-qwen. It is a CLI coding agent with a rich
TUI that can plan, edit, run, and verify code autonomously.

══ ARCHITECTURE OVERVIEW ══════════════════════════════════════

nedster/
├── nedster.py          ← Main CLI entry point (replaces main.py)
├── agent.py            ← Core agentic loop (replaces rag.py)
├── context_loader.py   ← Project context builder (NEW)
├── editor.py           ← Multi-file code editor with diff (NEW)
├── git_tools.py        ← Git integration (NEW)
├── code_tools.py       ← Lint, test, format runners (NEW)
├── tui.py              ← Rich TUI renderer (NEW)
├── tools.py            ← Extend existing Aria tools.py
├── memory.py           ← Keep existing Aria memory.py
├── retriever.py        ← Keep existing Aria retriever.py
├── journal.py          ← Keep existing Aria journal.py
├── personality.py      ← Adapt for Nedster persona
├── Modelfile           ← Aria Modelfile (already exists)
├── NEDSTER.md          ← Per-project memory file (like CLAUDE.md)
└── requirements.txt    ← Add: rich, pathspec, gitpython

══ STEP 1: nedster.py (CLI Entry Point) ══════════════════════

Build argparse CLI with these commands:

  nedster                        → interactive agent REPL
  nedster --project /path        → set working project directory
  nedster --auto                 → no confirmation, full auto mode
  nedster --think                → enable Qwen think mode
  nedster "fix the bug in foo.py" → one-shot task
  nedster init                   → create NEDSTER.md in cwd
  nedster stats                  → show context, VRAM, vector stats
  nedster reset                  → wipe ChromaDB

On startup:
  1. Check ollama is running (existing check_ollama())
  2. Load NEDSTER.md if present in cwd (project memory)
  3. Run context_loader.scan_project(cwd) — silent
  4. Print:  "Nedster. {project_name}. ({file_count} files | {vec_count} vectors)"
  5. Enter REPL loop

REPL slash commands:
  /add <file>     → manually add file to context
  /diff           → show pending diffs
  /apply          → apply all pending diffs
  /git            → git status + recent log
  /test           → run detected test suite
  /undo           → revert last file edit
  /clear          → clear short-term memory
  /stats          → VRAM + token budget + tool status
  /auto           → toggle auto-approve mode
  /think          → toggle think mode on/off

══ STEP 2: context_loader.py (Project Context Builder) ═══════

class ContextLoader:
    def __init__(self, project_dir: str):
        self.root = Path(project_dir)
        self.file_index = {}   # path → (size, mtime, language)
        self.active_files = [] # files in current context window
        self.gitignore_spec = None

    def scan_project(self):
        """
        Walk project dir. Skip: .git, __pycache__, node_modules,
        .venv, venv, *.pyc, *.so, *.bin, images, lock files.
        Use pathspec to honor .gitignore.
        Build self.file_index.
        Print: "[Nedster] Scanned {n} files in {project_name}"
        """

    def select_context_files(self, query: str, max_tokens=2000) -> list[str]:
        """
        Smart file selection for a given query:
        1. Semantic search via Retriever (existing ChromaDB)
        2. BM25 keyword match on file names + first 30 lines
        3. If query mentions a filename → always include it
        4. Always include: NEDSTER.md, main entry file, active_files
        5. Respect token budget (tiktoken count)
        Returns list of (filepath, content) tuples
        """

    def build_context_block(self, files: list) -> str:
        """
        Format: 
        === FILE: path/to/file.py ===
        <content (truncated if >150 lines)>
        === END FILE ===
        """

    def read_nedster_md(self) -> str:
        """Read NEDSTER.md project memory. Return '' if not found."""

    def update_nedster_md(self, new_facts: str):
        """
        Append new facts to NEDSTER.md under
        ## Session {datetime}
        Extracted by LLM at session end (like milestones.md).
        """

══ STEP 3: editor.py (Multi-File Code Editor) ═══════════════

class FileEditor:
    def __init__(self):
        self.pending_edits = {}  # path → (original, new_content)
        self.undo_stack = []

    def parse_edit_blocks(self, llm_response: str) -> list[dict]:
        """
        Parse LLM output for edit blocks in ALL these formats:

        Format A (preferred — NEDSTER native):
        <edit file="path/to/file.py">
        <old>exact old code</old>
        <new>replacement code</new>
        </edit>

        Format B (create/overwrite):
        <create file="path/to/new.py">
        full file content
        </create>

        Format C (bash output — treat as run_bash):
```bash
        command here
```

        Format D (markdown code with filename comment):
```python
        # path/to/file.py
        full content
```

        Return list of dicts:
        {"type": "edit|create|bash", "path": str,
         "old": str, "new": str, "content": str, "cmd": str}
        """

    def apply_edit(self, edit: dict, auto: bool = False) -> str:
        """
        For type=edit:
          - Read file, find exact `old` block, replace with `new`
          - If old not found: fuzzy match (difflib SequenceMatcher >0.8)
          - Store original in undo_stack
          - If auto=False: show colored diff, ask Y/n
          - Write file
          - Return: "Edited path/to/file.py (+5/-3 lines)"

        For type=create:
          - If file exists and auto=False: show diff, confirm
          - Write file
          - Return: "Created path/to/file.py (42 lines)"

        Use difflib.unified_diff for colored diff display.
        Green = additions (+), Red = removals (-).
        """

    def show_diff(self, path: str, original: str, new: str):
        """Print colored unified diff using rich or ANSI codes."""

    def undo_last(self) -> str:
        """Restore last edited file from undo_stack."""

══ STEP 4: git_tools.py ════════════════════════════════════

Functions (all return str for tool registry):

  git_status(cwd: str) -> str
    Run: git status --short + git log --oneline -5
    
  git_diff(cwd: str, file: str = "") -> str
    Run: git diff HEAD [file] | head -100

  git_commit(cwd: str, message: str) -> str
    Run: git add -A && git commit -m "{message}"
    Auto-generate message if empty: ask LLM for conventional commit msg

  git_branch(cwd: str) -> str
    Run: git branch --show-current

  git_stash(cwd: str) -> str
    Run: git stash

Add all to TOOL_REGISTRY in tools.py.

══ STEP 5: code_tools.py ════════════════════════════════════

Auto-detect and run project tools:

  detect_test_runner(cwd: str) -> str
    Check for: pytest.ini, pyproject.toml [tool.pytest], 
    package.json scripts.test, Makefile test target.
    Return: "pytest" | "npm test" | "make test" | "unknown"

  run_tests(cwd: str, file: str = "") -> str
    Run detected test suite. Capture output.
    Parse failures: extract "FAILED test_foo.py::test_bar" lines.
    Return: summary + failed test names

  run_linter(cwd: str, file: str = "") -> str
    Check for: ruff, flake8, eslint, mypy.
    Run on file or whole project. Return issues summary.

  run_formatter(cwd: str, file: str = "") -> str
    Check for: black, ruff format, prettier.
    Run and return: "Formatted 3 files"

  check_syntax(code: str, language: str = "python") -> str
    python: compile(code, '<string>', 'exec') — catch SyntaxError
    Return: "OK" or "SyntaxError line N: {msg}"

Add all to TOOL_REGISTRY.

══ STEP 6: agent.py (Core Agentic Loop) ════════════════════

class NedsterAgent (extends/replaces RAGPipeline):

  def __init__(self, project_dir: str, auto: bool, think: bool):
      self.context_loader = ContextLoader(project_dir)
      self.editor = FileEditor()
      self.retriever = Retriever()        # existing
      self.memory = MemoryManager()       # existing
      self.model = "aria-qwen"
      self.auto = auto                    # skip confirmations
      self.think = think
      self.tool_stats = {"calls":0,"loops":0,"edits":0,"tests":0}
      self._boot_project()

  def _boot_project(self):
      """
      1. context_loader.scan_project()
      2. Read NEDSTER.md → inject into system prompt
      3. Load milestones (existing logic)
      4. probe_tools() → _tool_inventory
      5. Print boot summary
      """

  def generate(self, user_input: str):
      """
      Full agentic loop:

      Phase 1 — CLASSIFY:
        Use classify_input() (existing). Also detect:
        - "code task": mentions file extension, edit, fix, write, implement
        - "git task": commit, branch, status, diff
        - "test task": test, run, check, failing

      Phase 2 — CONTEXT:
        If code/git/test task:
          files = context_loader.select_context_files(user_input)
          context_block = context_loader.build_context_block(files)
        Else: use RAG retrieval (existing)

      Phase 3 — PROMPT ASSEMBLY:
        system = NEDSTER_SYSTEM_PROMPT (see below)
        + NEDSTER.md contents
        + active tool inventory
        messages = [system] + memory.get_context_messages()
                   + [user: context_block + user_input]

      Phase 4 — GENERATE (streaming, existing logic):
        Stream response, strip <think> tags (existing).
        Collect full_response.

      Phase 5 — PARSE & EXECUTE (tool loop, max 5 iterations):
        A. Parse edit blocks via editor.parse_edit_blocks()
        B. Parse tool calls via tools.parse_tool_calls() (existing)
        C. For each edit: editor.apply_edit(edit, auto=self.auto)
           After edits: auto-run check_syntax() on changed files
           If syntax OK and test runner detected: offer to run tests
        D. For each tool call: execute via TOOL_REGISTRY (existing)
        E. Feed results back → continue loop
        F. WATCHDOG.ping() each iteration (existing)

      Phase 6 — POST:
        memory.add_turn(user_input, response) (existing)
        retriever.add_to_memory(...) (existing)
        Update tool_stats (track edits, tests run)
        Log to session file

  def plan_and_execute(self, task: str):
      """
      For complex multi-file tasks:
      1. Ask LLM to produce a numbered plan (existing plan logic)
      2. Show plan to user, confirm (unless auto mode)
      3. Execute each step in sequence
      4. After all steps: run_tests() + run_linter()
      5. git_status() to show what changed
      6. Offer git_commit() with auto-generated message
      """

══ STEP 7: NEDSTER SYSTEM PROMPT ════════════════════════════

Adapt Aria's 12 directives. Key changes:

DIRECTIVE ZERO: Language = English only.

DIRECTIVE ONE: You are Nedster, a local coding agent.
User is H2. You operate on their codebase.
You have these edit formats available:
  <edit file="path"><old>...</old><new>...</new></edit>
  <create file="path">...</create>
Use these for ALL file changes — never describe changes, make them.

DIRECTIVE TWO (CODE EDITS):
When asked to fix/implement/refactor:
  1. Read the relevant file first (if not in context)
  2. Make the minimal edit — not a rewrite unless asked
  3. Always emit <edit> or <create> blocks — never "you should change X"
  4. After edit: check_syntax() automatically
  5. Never leave TODOs or placeholder comments

DIRECTIVE THREE (PLANNING):
For multi-file tasks, emit a plan first:
  Step 1: [what] in [file]
  Step 2: [what] in [file]
  Awaiting approval.
On any poke ("ok", "go", "yes", "!") → execute all steps.

DIRECTIVE FOUR (GIT):
After completing a coding task:
  - Run git_status() silently
  - If changes exist and H2 hasn't mentioned git: offer one commit line
  - Never commit without mentioning it

DIRECTIVE FIVE (TOOL PRIORITY):
read_file → before editing any file not in context
run_bash  → for installs, builds, verification
run_tests → after edits that affect logic
git_*     → after successful task completion

DIRECTIVE SIX (NEDSTER.md):
On task completion, extract any project facts:
  architecture decisions, dependencies added, patterns used.
Append to NEDSTER.md silently. Say: "Project memory updated."

DIRECTIVE SEVEN (VRAM/HARDWARE):
H2 hardware: RTX 3060 Ti 8GB, i7-11700k, 64GB, Pop!OS.
aria-qwen context limit: 4096 tokens.
Keep context lean. Prioritize minimal file loading.

DIRECTIVE EIGHT (EXISTING ARIA DIRECTIVES):
Inherit: emotional calibration, no menus, no noise,
safe archive, memory rules, financial data rules.
(Full 12 directives from existing Modelfile apply.)

══ STEP 8: tui.py (Rich Terminal UI) ════════════════════════

Use `rich` library for:

class NedsterTUI:
  COLORS = {
    "prompt":   "bold cyan",      # Nedster> 
    "user":     "bold white",     # You:
    "agent":    "green",          # response text
    "tool":     "dim yellow",     # [tool: run_bash]
    "edit":     "bold blue",      # [edit: file.py]
    "success":  "bold green",     # [OK]
    "warning":  "bold yellow",    # ⚠️
    "error":    "bold red",       # [ERROR]
    "thinking": "dim italic",     # <think> content if visible
    "footer":   "dim",            # token count footer
  }

  def print_diff(self, path, original, new_content):
      """Colored unified diff with + green / - red lines."""

  def print_tool_call(self, tool_name, args):
      """[tool: bash] cmd → ...]  one line, dim."""

  def print_edit_preview(self, edit: dict) -> bool:
      """Show diff. Prompt Y/n. Return bool."""

  def print_response_stream(self, chunk: str):
      """Stream text to console in green."""

  def print_status(self, msg: str, style: str = "dim"):
      """[Nedster] status message"""

  def print_boot(self, project: str, files: int, vectors: int):
      """
      ┌─────────────────────────────────┐
      │  Nedster  ·  {project}          │
      │  {files} files · {vectors} vecs │
      └─────────────────────────────────┘
      """

══ STEP 9: NEDSTER.md Template ══════════════════════════════

When `nedster init` is run, create NEDSTER.md:

# NEDSTER Project Memory
## Project: {cwd_name}
## Language: (auto-detected)
## Entry Point: (auto-detected main.py / index.js / etc.)
## Test Runner: (auto-detected)
## Key Dependencies: (from requirements.txt / package.json)
## Architecture Notes:
(populated by agent over time)
## Decisions:
(populated by agent over time)
## Sessions:
(populated automatically)

══ STEP 10: requirements.txt additions ══════════════════════

Add to existing requirements:
  rich>=13.0.0
  pathspec>=0.12.0
  gitpython>=3.1.40
  difflib  (stdlib)

══ IMPLEMENTATION ORDER ═════════════════════════════════════

Build in this order (each step must be working before next):

1. tui.py — get rich output working first
2. context_loader.py — scan_project + select_context_files
3. editor.py — parse_edit_blocks + apply_edit with diff display
4. git_tools.py — all 5 functions + add to TOOL_REGISTRY
5. code_tools.py — detect_test_runner + run_tests + check_syntax
6. agent.py — full NedsterAgent class wiring everything together
7. nedster.py — CLI entry point + REPL loop
8. Test end-to-end: nedster --project ~/some_python_project
   Ask: "fix any type errors in main.py"
   Verify: edit block parsed, diff shown, file updated, syntax checked

══ QUALITY RULES ════════════════════════════════════════════

- Every new function must have a docstring
- All file I/O: use try/except, return error str (not raise)
- All subprocess calls: timeout=15, capture stderr
- Token budget always enforced (tiktoken check before LLM call)
- Secrets never logged (existing sanitize_output() applies)
- No prints inside library functions — use tui.print_status()
- Fuzzy edit matching: difflib.SequenceMatcher ratio > 0.75
- If edit old_block not found: report clearly, do not silently skip
- Auto mode: log every auto-approved action to session_memory.md

══ DELIVERABLE CHECK ════════════════════════════════════════

When done, run this self-check:

  python nedster.py init          → creates NEDSTER.md ✓
  python nedster.py stats         → shows VRAM + vectors ✓
  python nedster.py "list files"  → uses list_dir tool ✓
  python nedster.py "read main.py and summarize it"
                                  → context_loader loads file ✓
  python nedster.py "add a hello() function to main.py"
                                  → edit block emitted + applied ✓
  python nedster.py --auto "run the tests"
                                  → auto mode, no confirm ✓

════════════════════════════════════════════════════════════
START: Read combined3.txt (existing Aria codebase) first.
Reuse: memory.py, retriever.py, journal.py, tools.py, Modelfile.
Do NOT rewrite what already works — extend it.
Build Nedster on top of Aria's proven foundation.
════════════════════════════════════════════════════════════
