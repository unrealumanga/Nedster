# Nedster: a local coding agent that saves your Claude Code and Codex tokens

[![CI](https://github.com/unrealumanga/Nedster/actions/workflows/ci.yml/badge.svg)](https://github.com/unrealumanga/Nedster/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Windows | macOS | Linux](https://img.shields.io/badge/os-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![Runs on Ollama](https://img.shields.io/badge/LLM-Ollama%20(local)-black)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Your paid AI coding agent (Claude Code, Codex, Cursor, Gemini CLI, Aider...) is good at planning and reviewing.
It is expensive at grinding: re-reading files, edit-test-fix loops, renames, small fixes. Those loops eat your
token budget and your plan's usage limits.

**Nedster does the grinding on your own GPU.** Your paid agent writes a short brief, Nedster runs the
read-edit-test loop locally with an open-weights model through [Ollama](https://ollama.com), and hands back a
JSON result that says what changed and whether the tests passed. Your paid agent reviews the diff instead of
producing it.

```text
you ──▶ paid agent (plans, writes a ~500-token brief)
            │  nedster run --brief task.md --json
            ▼
        Nedster on your GPU (reads, edits, runs tests: ~50K tokens, $0)
            │  {"outcome": "completed", "changed_paths": [...], "reasons": [...]}
            ▼
        paid agent reviews the diff (~1-3K tokens) ──▶ you
```

Local, offline after setup, no account, MIT licensed. Windows, macOS and Linux.

## Measured, not promised

From the 14-task [companion eval suite](#evals) (realistic tasks: a bug from a traceback, a 3-file feature, an
8-file rename, a one-function edit in a 2,759-line file, a JavaScript fix, writing tests that must catch hidden
mutants...), two runs each, `qwen3.6:35b` (MoE, Q4) on an RTX 5090:

| Per delegated task | Nedster (local) | Your paid agent |
| --- | --- | --- |
| Prompt tokens processed | 48K median, 87K mean | ~500-650 to write the brief |
| Tokens generated | ~1.7K median | ~1-3K to read the JSON result and review the diff |
| Wall time | 10 s median, 14 s mean | - |
| Cost | electricity | a few thousand tokens instead of tens of thousands |

- **Result quality:** 23 of 28 runs passed the fixture's own hidden checks; 1 false success. On the 16 simpler
  toy fixtures: 16/16, 0 false successes.
- **Honest caveat:** the token figures are what Nedster's own loop consumed. A paid agent doing the same loop
  re-reads similar files and test output, but it also caches repeated input at a discount, so your *bill* drops
  by less than the raw token ratio. What you feel most is the **usage limit lasting longer**: the mechanical loops
  stop counting against it.
- **This repo is dogfooded that way.** Parts of Nedster's own latest release (CLI flag checks, `config get`,
  `doctor` advice, `prune --dry-run`, the progress line, test fixes) were implemented by Nedster from briefs
  written by Claude Code, then reviewed. The briefs it got are the template below.

## Quick start (5 minutes)

Requirements: Python 3.11+, [Ollama](https://ollama.com) (setup can install it), 8 GB RAM minimum. A GPU is
strongly recommended: see [Local models](#local-models-and-hardware).

```bash
git clone https://github.com/unrealumanga/Nedster.git
cd Nedster
./install.sh                                            # macOS / Linux
# powershell -ExecutionPolicy Bypass -File .\install.ps1   # Windows
```

The installer puts `nedster` on your PATH and runs `nedster setup`, which detects your hardware, picks and pulls a
model, and writes `~/.nedster/config.toml`.

Nedster only edits files until you let it run commands. For delegation it must run your tests:

```bash
nedster setup --allow-execution         # run tests only inside a sandbox (Docker / bubblewrap / Seatbelt)
nedster setup --allow-host-execution    # or: run directly on this machine, no isolation
nedster doctor                          # must say: inference ready, execution ready
```

Try it once by hand:

```bash
cd your-project
nedster run "Find the failing test, fix the cause, and run the suite"
nedster                                  # or open the terminal UI
```

## Use it with Claude Code

**1. Tell Claude Code that Nedster exists.** Add this to your project's `CLAUDE.md` (or `~/.claude/CLAUDE.md`
for all projects):

````markdown
## Local worker: Nedster (saves tokens)
`nedster` is a local coding agent on this machine (Ollama, free). Delegate mechanical coding to it.

Delegate: bug fixes with a known location or traceback; 1-3 file features with tests; renames; one change in a
large file; writing tests for existing code; "where is X / how does Y flow" questions.
Do yourself: unclear design, greenfield modules from a spec, anything that deletes or moves files, security code.

How to delegate:
1. Write a brief to `.nedster/briefs/<name>.md` (template: exact files, shape of the change, what must not
   change, the test command as an argv list).
2. Run `nedster run --brief .nedster/briefs/<name>.md --json` (can take a few minutes; run it in the background
   for long tasks).
3. Read `outcome`, `changed_paths`, `reasons` from the JSON. Exit code 0 = completed or answered.
   `unvalidated` = files changed but no passing test run. `incomplete` = not done; read `reasons`.
4. Before accepting: review `git diff` and run the tests yourself. Never trust the label alone.
5. If it fails twice on the same brief, do the task yourself.
Questions: `nedster run --role explain --expect answer "<question>" --json` and read only the files it cites.
````

Add `.nedster/briefs/` to `.gitignore` if you do not want to keep the briefs.

**2. Ask for it in plain words.** Prompts that work:

```text
Use Nedster for this: fix the KeyError in billing.parse_rates (traceback below). Write the brief, run it,
then review the diff and rerun the tests yourself.

Plan the auth-refactor yourself, then split it into Nedster briefs of one or two files each. Run them one at a
time and review each diff before the next.

Before you read the codebase, ask Nedster (role explain) where the retry logic lives and how a webhook reaches
the queue. Only open the files it names.

Have Nedster add tests for utils/intervals.py that cover overlap, touching and empty intervals. Check that the
tests fail if you break merge().
```

Claude Code runs Nedster through its Bash tool. A task usually finishes in seconds to a few minutes; for longer
ones ask Claude Code to run it in the background.

## Use it with Codex (and other agents)

Codex reads `AGENTS.md`. Put the same block as above in your project's `AGENTS.md` (Nedster reads that file too,
as its own project instructions, which is fine: the rules apply to both).

Nedster talks to Ollama at `http://localhost:11434`. Codex's default workspace sandbox blocks network access, so
either approve the `nedster` command when Codex asks, or allow network access for the workspace sandbox in Codex's
config (`sandbox_workspace_write.network_access = true`).

The same pattern works for any agent that can run a shell command: Cursor's agent, Gemini CLI, Aider (`/run`),
Copilot CLI, or your own scripts.

## Use it from scripts

```bash
nedster run --brief task.md --json > result.json; echo "exit $?"
jq -r '.outcome, .changed_paths[], .reasons[]' result.json
```

| Exit code | Meaning |
| --- | --- |
| 0 | `completed` (changed and a test run passed after the last change) or `answered` (a question) |
| 1 | anything else: `unvalidated`, `incomplete`, `no_change_needed`, `cancelled`, `failed` |
| 2 | usage error (bad flags, unknown session, empty brief) |
| 3 | not ready (no model server, nothing installed) |

`--json` prints one document (`outcome`, `reasons`, `changed_paths`, `answer`, `session_id`). `--jsonl` streams
every engine event, one per line. `--role plan|review|explain` makes a run read-only, enforced at tool dispatch:
use it when you only want an answer, a plan or a review.

## Writing a brief that works

The difference between a brief and a wish: Nedster with a terse one-line request passed 22-23 of 28 realistic
tasks; with briefs like this one, the tasks it failed (false premises, vague cleanups, unfamiliar library APIs)
passed 6/6.

```markdown
<One sentence: what to change and why.>

Exact changes:
1. <path>: <what changes; give the signature, field lines or data structure when correctness depends on it>
2. <path>: ...
3. Test: <existing test file to extend, or tests/test_<x>.py with these cases: ...>

Do not change: <files or dirs>. Do not delete or move any file.
If <the function / setting / file you name> does not exist, say so and change nothing.
Use only <library calls>; do not add options I did not ask for.

Rules: commands are argv lists, e.g. ["python", "-m", "pytest", "-q", "tests/test_x.py"].
After an edit, use the new sha from that edit's result for the next edit of the same file.

Done when ["python", "-m", "pytest", "-q", "tests/test_x.py"] passes. Run it.
```

A real one (19 s; the logic was right, review removed some redundant code):

```markdown
Make the engine's step and tool-call limits configurable. Today ironclad/native.py builds
Budget(context_tokens=...), so max_steps is always 40.

Exact changes:
1. ironclad/config.py, class RuntimeConfig: add `max_steps: int = 40` and `max_tool_calls: int = 120`.
2. ironclad/native.py: add `engine_budget(config, model) -> Budget` (0 means no limit) and use it where the
   Engine is built.
3. tests/test_engine_budget.py: defaults; 80 -> 80; 0 -> None; NEDSTER_RUNTIME__MAX_STEPS=75 via load_config.

Done when ["python", "-m", "pytest", "-q", "tests/test_engine_budget.py"] passes. Run it.
```

**Do not delegate** (measured failures, not guesses):

| Request | What happened | Instead |
| --- | --- | --- |
| "Clean up this repo" | Deleted backups and notes in every run | List exactly what may be deleted, or do it yourself |
| "Change the retry_backoff setting" (it does not exist) | Added a new setting instead of saying so | Add "if it does not exist, say so and change nothing" |
| "Implement this module from SPEC.md" | Its own new tests passed, the spec was still violated | Write the tests yourself first, or keep greenfield work |
| Five interdependent new files in one brief | Ran out of its step budget | One deliverable per brief |

## Local models and hardware

Nedster runs one open-weights model through Ollama. `nedster setup` picks a plan from your hardware
(`nedster models recommend` shows it):

| Hardware | fast | balanced | maximum |
| --- | --- | --- | --- |
| ~8 GB VRAM/RAM | qwen3.5:4b | qwen3.5:9b | qwen3.5:9b, larger context |
| ~16 GB | qwen3.5:9b | qwen3.5:27b | qwen3-coder:30b |
| ~32 GB | qwen3.5:9b | qwen3.6:27b | qwen3.6:35b |

The numbers on this page come from the top tier (`qwen3.6:35b`, 64K context, RTX 5090: prompt processing about
4,600 tokens/s, generation about 215 tokens/s). Smaller models are slower per task and pass fewer of the harder
fixtures; run the [evals](#evals) on your machine to see your own numbers before you rely on it.

What matters for speed, from measurements:

- **Keep VRAM headroom.** Leave at least ~3 GB of GPU memory free. With a 3D/rendering app holding 9 GB next to
  the model, Windows paged GPU memory and Nedster got 15-100x slower (generation fell from ~300 to ~9 tokens/s)
  with no error. `nvidia-smi` shows it; closing the app fixed it.
- **One job at a time per GPU.** Two Nedster runs (or Nedster plus another model) evict each other's prompt
  cache: a warm turn takes 0.2 s, a cold re-read of a 56K-token history ~10 s.
- **Quantize the KV cache.** Set `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` for the Ollama server
  and restart it: about twice the context in the same VRAM. `nedster doctor` prints the exact command for your OS.
- **Context follows the model.** Nedster reads the model's real window (`num_ctx`) from Ollama and compacts old
  tool output deterministically, so long sessions keep working without sending oversized requests.

Pick or switch models with `nedster models` / `nedster models use <name>`, or F4 in the terminal UI.

## Why trust its results

- **Outcomes come from tool records, not from the model's own claims.** `completed` means the files changed and
  a recognized test runner (pytest, unittest, npm, node --test, cargo, go) passed after the last change. A change
  request that changed nothing ends `incomplete`. Changes without a passing test run end `unvalidated`.
- **Auto-verify:** if the model edits and stops without testing, Nedster runs the tests itself.
- **Guarded edits:** an edit needs the file's current hash from a read; stale edits are refused; writes are atomic;
  `/undo` restores the last turn's edits.
- **Consent and sandboxes:** commands run only after you allow them, in Docker / bubblewrap / Seatbelt, or on the
  host if you opt in. A cloned repo's config cannot loosen any of that.
- **Read-only roles** (`plan`, `review`, `explain`) are enforced at tool dispatch, not just in the prompt.

## Using it on its own

Nedster is also a complete terminal coding agent. `nedster` opens the TUI: chat, `@file` completion, named
sessions with a picker (Ctrl+S), model picker (F4), plan mode (Shift+Tab, then `/act`), Ctrl+R history search,
`/undo`, `/diff`, a status line with context left, tokens/s and step count. `nedster init` writes a starter
`AGENTS.md`; an opt-in personality (`runtime.soul`) adds dry status lines.

Full reference (every command, key, setting, sandbox detail, the eval tables): **[docs/USAGE.md](docs/USAGE.md)**.

## Evals

Two offline suites with check scripts that decide pass/fail independently of the agent:

```bash
python -m evals.run --model nedster-worker                                   # 16 toy fixtures
python -m evals.run --root evals/companion --model nedster-worker --repeat 2  # 14 realistic tasks
```

Current results and per-fixture tables: [docs/USAGE.md#evals](docs/USAGE.md#evals). Run them on your hardware
and model before deciding what to delegate.

## FAQ

**Does it replace Claude Code or Codex?** No. A 35B local model is a good worker and a poor architect. Keep the
paid agent for design, ambiguous requests and review; give Nedster the well-specified loops.

**Does anything leave my machine?** No. Inference goes to your local Ollama server; no account, no telemetry, no
cloud fallback.

**Which local models work?** Any Ollama model with tool calling. The defaults are Qwen 3.5 / 3.6 sizes picked
for your hardware; `nedster models use <name>` switches.

**No GPU?** It runs on CPU with small models, slowly. Delegation pays off with a GPU that fits a 9B+ model.

**How do I know it worked?** Read `outcome` and `reasons`, then review the diff and run the tests. The label is
evidence, not a guarantee: `completed` means the tests that ran passed, which is only as good as the tests.

## Development

```bash
python -m pip install -e ".[dev]" && python -m pytest -q tests
python -m ruff check nedster_agent ironclad tests
```

Design notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Security model: [SECURITY.md](SECURITY.md).
Upgrading from v3: [docs/USAGE.md#upgrading-from-v3](docs/USAGE.md#upgrading-from-v3). MIT licensed.
