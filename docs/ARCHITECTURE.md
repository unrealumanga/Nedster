# Nedster architecture (v4)

Nedster is a local-first coding agent: one native tool-calling loop, a typed tool runtime with policy and
sandbox backends, a SQLite session store, and a streaming transport to Ollama. The contracts between the parts
live in `ironclad/contracts.py`; the engine owns changes to it.

```text
 nedster CLI / TUI  (nedster_agent/cli.py, tui.py, widgets/)
        |  EngineEvent stream  <------------------------------------------+
        v                                                                  |
 NativeSession (ironclad/native.py)  -- wires config -> parts, checkpoints, undo
        |
        v
 Engine loop (ironclad/engine/loop.py)
   prompt.py  intent.py  state.py  context.py  acceptance.py
        |   chat(messages, tool specs)            |  tool calls
        v                                         v
 Transport (ironclad/transport/)         Tool runtime (ironclad/tool_runtime/)
   ollama.py  openai_compat.py             CompositeToolRuntime: schema -> policy -> run
   fake.py (tests)                           files.py search.py git.py exec.py
        |                                             |
        v                                             v
 Ollama /api/chat                         backends: bwrap | seatbelt | docker | host
                                          (Windows Job Object / POSIX process group)
        ^                                             |
        |       ToolRecord(call, result)              |
        +---- SessionStore (ironclad/session_store/) <+
              sqlite_store.py: messages, outputs, state, transcript, locks
              checkpoints.py : per-turn file pre-images for /undo
```

## Principles

1. **Facts come from tool records.** Every executed call becomes a `ToolRecord(call, result)`. Acceptance, compaction
   summaries and the final report are computed from records. Model prose never upgrades an outcome.
2. **The model proposes, the runtime decides.** Arguments are validated against each tool's JSON schema; policy is
   evaluated on the *resolved* argv; execution happens only in a selected backend.
3. **Fail closed.** No backend, no consent, a stale hash, a protected path or a failed journal write all refuse the
   action with a status the model can read (`INVALID`, `DENIED`) instead of degrading.
4. **Append-only history between compactions**, so the model server's prefix cache stays valid.

## Engine (`ironclad/engine/`)

`loop.py` runs one conversation. Message layout: `[system prompt, pinned task state, *history]`.

Per run: classify the objective, then repeat *compact if needed -> check the window fits -> model call -> execute
tool calls -> record* until the model answers without tool calls or a stop condition fires (cancel, deadline, step /
token / tool-call budget, context exhausted, model/transport error). Truncated model turns (`length`, `error`) never
execute their tool calls. A call repeated unchanged 3 times stops the task. Calls pair with results in every
request; `repair_history` closes unpaired calls after a crash.

- **Prompt** (`prompt.py`): fixed instructions plus project facts computed once per session and persisted, so the
  text is byte-identical across turns.
- **Intent** (`intent.py`): `classify(objective)` -> `change | question | unknown` by a deterministic sentence/clause
  heuristic. The CLI `--expect change|answer` overrides it. It is stored in the task state so a resumed session
  keeps it.
- **Task state** (`state.py`): objective, intent, files touched, and other facts, rendered as one pinned message.
- **Auto-verify**: if the model changed files and stopped without validating, the engine issues a `run_tests`
  call itself (max 2 per run; a failure gives the model another turn).

### Acceptance (`acceptance.py`)

`evaluate(records, stop, detail, intent)` returns an `Outcome` from records only:

- `COMPLETED`: a change succeeded and a *trusted* validation passed after the latest change. Trusted = `run_tests`,
  or a flagged `run_command` that ran a recognized runner with a known test count > 0. Unknown or zero test
  counts, stale validations (a mutation or revision bump after them), a baseline run that was already green before
  the change, and runs during which files changed do not count.
- `ANSWERED`: a final run with no change where intent is not `change`.
- `INCOMPLETE`: intent is `change` but no change succeeded ("requested change was not made"), validation failing,
  or budget/context stop.
- `UNVALIDATED`: changes without trusted passing validation.
- `NO_CHANGE_NEEDED` (a claim): intent `change`, no change made this run, a successful `declare_no_change`, at
  least one successful read, and a passing trusted run **after** the last read with nothing mutating or failing
  after it. The model's reason text is shown but is never evidence.
- `CANCELLED`, `FAILED` from the stop condition.

### Context (`context.py`)

The window is `runtime.context_tokens`, or the served `num_ctx` (fallback 32768). A completion reserve
(budget/8, 512..4096 tokens) is always left free. Before each call:

1. Soft trigger at 70% of the window: **mask** old tool outputs with stubs (`[output #N ...]`) that keep the
   `tool_call_id`; full text stays in the store, retrievable through `show_output`.
2. Still too large: **drop whole oldest assistant+result groups**, replaced by one deterministic summary message
   (reads and changes with sha, validations fresh/stale, failures, output handle ranges) built only from records,
   aiming at 50% of the window. The newest group and all user messages are kept verbatim.
3. Still not fitting: stop `INCOMPLETE` ("context budget exhausted"); no oversized request is sent.

Compaction is a discrete event; its counter and summary persist in session state across restarts. Rewriting early
history costs a full prefill on hybrid-attention models (see README, KV cache), so compactions are rare.

## Tool runtime (`ironclad/tool_runtime/`)

`CompositeToolRuntime` (`composite.py`) exposes the tool specs, validates arguments (`schema.py`), authorizes the
*resolved* program, starts jobs and bumps a workspace revision after any execution or mutation.

- **Files** (`files.py`): `read_file` (line ranges, returns sha256), `write_file`, `edit_file` (exact `old`/`new`
  or `edits[]`, atomic, `expected_sha` required, CRLF preserved, size cap), `list_files`. Protected paths
  (`.git`, `.nedster`, resolved gitdir targets) are write-denied by lexical and resolved checks. An identical
  write is a no-op. Writes run under a mutation lock.
- **Search** (`search.py`): `search_text` via a trusted `rg`, else a contained pure-Python fallback (regular
  files, inside the root).
- **Git** (`git.py`): `git_status/diff/log/show` read-only through a hardened git (no external diff, textconv,
  fsmonitor; hooks dir outside the sandbox-writable temp).
- **Execution** (`exec.py`): `run_command` and `run_tests`, argv arrays only, no shell. Programs are resolved from
  trusted absolute PATH entries outside the project. The child gets an allowlisted environment (no secrets).
  `run_tests` accepts only a recognized runner (pytest, `python -m unittest`, npm test, cargo test, go test,
  `node --test`) and parses only that runner's summary format. Output is bounded; cancel kills the process tree.
- **Backends** (`backends/`): `select_backend` tries `bwrap`, `seatbelt`, `docker` in that order, then `host` only
  with `sandbox.allow_host_execution`. An explicit backend that is not ready is an error, not a fallback. `winjob.py`
  contains Windows host processes with a Job Object (ctypes, kill-on-close); it limits processes, not file/network
  access. Details of what each isolates: README.
- **Policy**: `policy.max_auto_risk` (default `write`; `execute` needs consent) and `policy.allowed_commands`.
  `denied_fragments` is advisory substring matching, not a boundary.

## Config (`ironclad/config.py`)

Precedence: defaults < user `~/.nedster/config.toml` < project `.nedster/config.toml` < `NEDSTER_*` env < CLI.
The project layer is untrusted: only a small allowlist of keys is honoured, security keys can only be tightened,
everything else is ignored with a warning. `engine = "legacy"` warns and runs native.

## Session store (`ironclad/session_store/`)

SQLite in WAL mode, writes under `BEGIN IMMEDIATE`. Tables: `sessions`, `messages` (model history, rewritten on
compaction), `outputs` (full tool output by handle), `state` (task state, compaction state), `transcript` (display
history, append-only, never rewritten by compaction), `session_locks`. Sessions are bound to the canonical project
root. A lock (heartbeat + pid liveness) stops `delete` and `/resume` on a session another process is using.

- **Transcript**: what `/resume` and `sessions show` display. Tool entries keep name and status, not output.
  `/resume` re-executes nothing and the resumed evidence is not treated as current.
- **Checkpoints** (`checkpoints.py`): `FileTools` journals a file's pre-image before its first mutation in a turn.
  `end_turn` records post-image hashes. `undo_last` preflights every path (current hash = post hash, inside root,
  not protected, no symlink/junction swap) and restores nothing if any conflicts; restores are atomic per file
  and journaled so an interrupted undo is rolled forward or back at the next start. Open turns left by a dead
  process are closed by `recover()` (owner = host, pid, start time, nonce). Commands are not journaled; a turn
  that ran commands is reported as not fully covered.

## Transport (`ironclad/transport/`)

`ChatTransport` streams `ChatEvent`s: text deltas, one `ToolCallsReady`, then `TurnFinished` with usage taken from
the server's timings. `ollama.py` keeps one pooled `httpx` client and streams NDJSON from `/api/chat` with tools; it
never sends the model's thinking back. `openai_compat.py` is an SSE adapter for OpenAI-compatible servers; it is not wired into the CLI or
config in v4, so nothing selects it. `fake.py` scripts conversations for offline tests.

## Front ends

- `nedster_agent/cli.py`: parser built lazily (no heavy imports on `--help`); `run` emits plain text, `--json` or
  `--jsonl` (EngineEvents); `sessions` exit codes: usage 2, runtime error (e.g. active-lock refusal) 1. Setup/doctor/models logic is in
  `model_manager.py` (transactional config write with backup; readiness tiers inference / read-only / execution).
- `nedster_agent/tui.py` + `widgets/`: renders the EngineEvent stream, queues input while busy, Esc cancels, shows
  model/context/sandbox in the status line, runs the slash commands.

## Non-goals

No multi-agent router/planner/critic, no automatic cloud fallback, no MCP client or server in v4, no worker-to-worker
messaging, no hardware-rooted security claim, no automatic destructive actions. The v3 code remains on tag `v3-legacy`.
