# NEDSTER // IRONCLAD

**A local-first router/worker coding agent with capability-gated tools, hardware-aware model scheduling, verifiable state, and a live mission-control TUI.**

[![Ironclad quality](https://github.com/unrealumanga/Nedster/actions/workflows/quality.yml/badge.svg)](https://github.com/unrealumanga/Nedster/actions/workflows/quality.yml)
![Python](https://img.shields.io/badge/Python-3.11–3.13-72F1B8)
![Local first](https://img.shields.io/badge/default-local--only-7C5CFC)
![Status](https://img.shields.io/badge/status-3.0_alpha-FFB86C)

Nedster Ironclad is the 2026 evolution of the project: a small resident router plans work, role-specific local workers execute bounded manifests, a separate critic checks the evidence, and the Python runtime—not model text—owns every side effect.

The classic `nedster.py` interface remains available while Ironclad matures. It now uses a bounded multi-turn tool loop, honest replay outcomes, stall detection, one canonical `nedster` model tag, and a graded command policy. The new runtime is intentionally marked **3.0 alpha**; its security claims below distinguish implemented controls from roadmap work.

## Why this architecture now?

One giant model is a poor local-agent operating system. It consumes the entire memory budget, handles cheap routing expensively, and concentrates planning, execution, and judgment in one fallible context. Ironclad uses an asymmetric swarm instead:

```text
                         untrusted prompt / repository
                                      │
                         ┌────────────▼────────────┐
                         │ resident router model   │
                         │ typed plan + tool scope │
                         └────────────┬────────────┘
                                      │ manifests
              ┌───────────────────────┼──────────────────────┐
              ▼                       ▼                      ▼
      coder / reasoner         read-only workers          critic
       paged on demand          safe concurrency       evidence review
              └───────────────────────┬──────────────────────┘
                                      ▼
                       capability + policy boundary
                                      │
                           sandboxed tool execution
                                      │
                        HMAC hash-chained event ledger
```

What is embedded today:

- **Schema-constrained planning and tool proposals.** Ollama structured outputs replace brittle XML and regex intent parsing.
- **Hardware-aware fleet selection.** Nedster probes NVIDIA, ROCm, Apple unified memory, RAM, and available local models before assigning router, coder, reasoner, and critic roles.
- **Asymmetric residency.** The router uses an infinite Ollama keep-alive; worker models use a short lease so VRAM can be reclaimed.
- **Safe swarm concurrency.** Independent read-only steps can run concurrently. Any write or execution step is serialized.
- **Runtime-owned authority.** Each action is normalized, policy-checked, then receives a short-lived, request-bound, single-use HMAC capability. Models never mint permissions.
- **Project confinement and isolation.** File paths are resolved beneath the selected project. Linux prefers Bubblewrap, gVisor, rootless Podman, or Docker; network access is denied by default.
- **Verifiable state.** Every objective, plan, decision, action, and result enters a local SQLite WAL with an HMAC hash chain.
- **Modern inference profiles.** `llama-command` emits a loopback-only llama.cpp launch plan with mmap, Flash Attention, quantized KV cache, and prompt-lookup or draft-model speculative decoding.
- **Optional GPT-5.6 Sol oracle.** High-value reasoning can be escalated to the Responses API only when explicitly enabled. Ordinary work remains local.

## Mission-control TUI

```text
╭─────────────────────────╮ ╭────────────────────────────────╮ ╭────────────────────────╮
│ MISSION GRAPH           │ │ LIVE COGNITION + EVENT STREAM  │ │ MODEL FLEET            │
│ 01 inspect      DONE    │ │ ◆ plan.created                 │ │ ROUTER   qwen…  ollama │
│ 02 implement    RUNNING │ │ ◆ tool.authorized              │ │ CODER    code…  ollama │
│ 03 verify       PENDING │ │ ◆ step.finished                │ │ CRITIC   small… ollama │
│                         │ │                                │ ├────────────────────────┤
│                         │ │ Describe the outcome…          │ │ RTX / RAM / VRAM       │
╰─────────────────────────╯ ╰────────────────────────────────╯ ╰────────────────────────╯
 SAFE-ZONE // titan-32g // sandbox=docker // ready
```

The Textual interface shows the live task DAG, model fleet, security events, critic outcomes, CPU/RAM/VRAM telemetry, and current sandbox posture. Commands: `/fleet`, `/ledger`, `/clear`, `/quit`.

## Quick start

### Linux (recommended)

```bash
git clone https://github.com/unrealumanga/Nedster.git
cd Nedster
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp config/ironclad.toml.example config/ironclad.toml

ollama serve
# Install at least one modern tool-capable/chat model that fits your machine.
ollama list

nedster-ironclad doctor
nedster-ironclad tui
```

Install Bubblewrap (`bwrap`) or rootless Podman for command isolation. Ironclad refuses an unsafe process fallback by default.

### Windows

```powershell
git clone https://github.com/unrealumanga/Nedster.git
Set-Location Nedster
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item config\ironclad.toml.example config\ironclad.toml
nedster-ironclad doctor
nedster-ironclad tui
```

Docker Desktop provides the strongest currently supported Windows execution boundary. Without a container backend, read/write file tools remain project-confined, but command execution stays disabled unless you deliberately pass `--unsafe-process`.

### macOS

Use the Linux steps, substituting a Homebrew Python if needed. Apple Silicon is treated as unified memory and receives a conservative 72% scheduling budget.

## Run modes

```bash
# Full-screen console
nedster-ironclad --project /path/to/repo tui

# One mission, read-only by default
nedster-ironclad --project . run "map the authentication flow and identify risks"

# Allow writes up to the configured risk ceiling
nedster-ironclad --project . --autonomous run "fix and test the parser"

# Inspect hardware, runtimes, and the selected fleet
nedster-ironclad doctor
nedster-ironclad models

# Verify that the event history still matches its HMAC hash chain
nedster-ironclad verify-ledger

# Print a tuned llama.cpp server command; nothing is launched
nedster-ironclad llama-command --model /models/worker.gguf
nedster-ironclad llama-command --model /models/worker.gguf --draft-model /models/router.gguf
```

## Hardware tiers

These are scheduling envelopes, not promises that every architecture will fit. Context length, MoE expert layout, KV type, GPU offload, and concurrent slots all change real memory use.

| Tier | Resident router | Paged worker envelope | Default concurrency | Inference profile |
|---|---:|---:|---:|---|
| Edge / 8 GB | 1.5–4B | 7–9B | 1 | Q4 weights, Q4 KV, 32K context |
| Forge / 16 GB | 3–8B | 12–20B | 2 read-only | Q5 weights, Q5 KV, 64K context |
| Titan / 32 GB | 7–14B | 27–40B | 3 read-only | Q6/8-bit weights, Q8 KV, 128K context |

With Ollama, Nedster scores the models already installed rather than silently downloading a fashionable checkpoint. Explicit model names in `config/ironclad.toml` always win.

## Optional GPT-5.6 Sol oracle

The cloud path uses OpenAI's Responses API with structured output, configurable reasoning effort, current-turn reasoning context, and `store=false`. It is off by default.

```bash
export OPENAI_API_KEY="..."                 # PowerShell: $env:OPENAI_API_KEY="..."
nedster-ironclad --oracle run "review this migration plan"
nedster-ironclad --oracle --pro run "analyze the hardest invariant"
```

Enabling the oracle changes the privacy boundary: the selected prompt and evidence are sent to OpenAI. `--pro` is a reasoning mode, not a separate model slug, and should be enabled only when your evaluations justify the extra latency and cost.

## Security posture

Ironclad follows one rule: **workers propose; the runtime authorizes**.

- Safe default: only read operations are automatically authorized.
- `--autonomous`: permits the configured ceiling, which defaults to non-destructive writes.
- Destructive or external actions remain outside the automatic ceiling.
- Tool commands use argv arrays, `shell=False`, and an executable allowlist.
- Workers do not receive signing secrets, capability keys, or direct tool handles.
- The RPC feature of llama.cpp is not used as a trust boundary; upstream documents it as insecure.

Capabilities are not a hardware enclave. A compromised host can inspect process memory. Firecracker lifecycle management, native Landlock rules, and remote attestation are roadmap items—not current claims. Read [SECURITY.md](SECURITY.md) and [the architecture notes](docs/ARCHITECTURE.md) before processing hostile repositories.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check ironclad tests
python -m compileall -q ironclad
```

CI runs on Linux, Windows, and macOS with Python 3.11 and 3.13. The classic interface is still available through `python nedster.py repl` for users of the existing plugin/RAG stack.

Existing installations can migrate legacy state and Ollama tags without deleting the original data:

```bash
DRY_RUN=1 ./scripts/migrate_from_aria.sh
REMOVE_LEGACY_TAGS=1 ./scripts/migrate_from_aria.sh
```

The migration is idempotent, preserves a timestamped backup, and moves supported state to `~/.nedster` without leaving an obsolete compatibility symlink.

## What comes next

The next meaningful leap is not another larger monolith. It is **stateful local inference infrastructure**: shared prefix/KV snapshots, topology-aware worker placement across GPU/CPU/NVMe, eval-driven routing, native Linux Landlock enforcement, reproducible micro-VM images, and privacy-budgeted oracle escalation. See [the August 2026 roadmap](docs/ROADMAP_2026.md).

Primary technical references: [OpenAI GPT-5.6 guidance](https://developers.openai.com/api/docs/guides/model-guidance?model=gpt-5.6), [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs), [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling), [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [llama.cpp speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md), [Linux Landlock](https://docs.kernel.org/userspace-api/landlock.html), [gVisor](https://gvisor.dev/), and [Firecracker](https://firecracker-microvm.github.io/).

## License

MIT © 2026 unrealumanga
