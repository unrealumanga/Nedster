# Nedster

**A persistent local AI coding agent for Ollama—with project-aware RAG, sandboxed tools, web research, and a concise terminal UI.**

Nedster turns an Ollama model into a practical terminal agent: it inspects local projects, retrieves relevant context, edits files, runs commands, follows compiler and test diagnostics, and verifies the real behavior before declaring success. The guided installer detects your GPU and RAM, installs or adopts Ollama, downloads a suitable model fleet, and creates a global `nedster` command.

> Status: **3.1 alpha.** The local coding loop works today. The setup experience and general-purpose integrations are being expanded quickly; see [what works now](#what-works-now).

## Why Nedster

- **Local by default.** Project context and routine inference stay on your computer. A cloud oracle is optional and never enabled merely because a key exists.
- **Setup that understands hardware.** Nedster detects NVIDIA, AMD, Apple Silicon, system RAM, and available VRAM before recommending Fast, Balanced, or Maximum models.
- **Persists until the result is verified.** Failed commands become diagnostic evidence, changed states can be retested, and bounded alternate branches replace repetitive retries.
- **Better results from a harness, not only a bigger model.** Structured planning, local RAG, typed tool calls, constrained execution, evidence grounding, and deterministic verification spend tokens where they matter.
- **Answers before it gives up.** Nedster can inspect accessible local files, run project tools, and use public-web search or fetch when current external facts are needed.
- **One familiar command.** Type `nedster` in any terminal. The current folder becomes the project; `nedster run "…"` handles one-shot automation.
- **Model roles instead of VRAM chaos.** A small router stays responsive while a stronger worker is loaded for coding and reasoning. Stable aliases let models change without rewriting the agent.
- **Extensible.** Nedster includes MCP server foundations and tool adapters. Creative apps and 3D workflows are valuable extensions—not a requirement for using the agent.

## Install

Requirements: Windows, macOS, or Linux; Python 3.11+; roughly 8 GB RAM minimum. A supported GPU is recommended but not required.

### Windows

```powershell
git clone https://github.com/unrealumanga/Nedster.git
cd Nedster
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

### macOS and Linux

```bash
git clone https://github.com/unrealumanga/Nedster.git
cd Nedster
chmod +x install.sh && ./install.sh
```

The installer:

1. installs the global `nedster` terminal launcher;
2. detects the machine and proposes a model profile;
3. keeps an existing Ollama installation or installs Ollama from its official source;
4. downloads the selected worker, router, and embedding models;
5. creates `nedster-worker` and `nedster-router` aliases with tuned context settings;
6. runs a local generation check and writes `~/.nedster/config.toml`.

It does not remove unrelated Ollama models or silently change global Ollama settings. Use `nedster setup --dry-run` to preview every action without downloading or writing.

## Use it from anywhere

Open a terminal in any repository or folder:

```text
cd my-project
nedster
```

Useful commands:

```bash
nedster run "Find the failing tests, fix the cause, and verify the suite"
nedster doctor
nedster doctor --plain
nedster permissions --plain
nedster models recommend
nedster models list
nedster setup --profile fast
```

Running bare `nedster` starts a quiet, chat-first TUI designed for 80-column terminals. Normal conversation shows the answer instead of streaming internal event names; press `F2` when you want execution details. The input stays visually stable while focused. Use `/help`, `/models`, `/doctor`, `/permissions`, `/clear`, and `/quit`.

When output is redirected or piped, Nedster does not open the full-screen TUI. Use `--plain` for stable one-record-per-line text, `--json` for structured output, `--no-tui` to force the linear path, `--no-color`/`--color` for an explicit color preference, and `--ascii` for restricted terminals.

The TUI is designed for a monospace font of **15 px or larger** at a normal desk distance; 18–20 px is recommended. Below 14 px, secondary text may be difficult to read. Full-screen terminal accessibility with screen readers is currently untested; the `run`, `doctor`, `models`, `setup`, `--plain`, and `--json` paths are the supported linear interfaces for assistive technology and automation.

## Automatic model matching

The defaults leave headroom for context, desktop graphics, tools, and concurrent work instead of filling every byte of VRAM.

| Hardware tier | Balanced worker | Responsive router | Typical use |
| --- | --- | --- | --- |
| Edge / CPU / ~8 GB | `qwen3.5:9b-q4_K_M` | `qwen3.5:4b-q4_K_M` | General tasks with modest context |
| Forge / ~16 GB | `qwen3.5:27b-q4_K_M` | `qwen3.5:4b-q4_K_M` | Stronger reasoning with model paging |
| Titan / ~32 GB | `qwen3.6:27b` | `qwen3.5:4b-q4_K_M` | Tool use, coding, reasoning, and large context |

Choose `fast` for low latency or `maximum` when quality matters more than model-loading delay. Recommendations are configuration, not lock-in: edit `~/.nedster/config.toml` or rerun setup at any time.

## The harness advantage

A useful local agent is more than a chat model:

```text
request → inspect → retrieve → structured plan → constrained tools → verify
                                      ↘ evidence ledger ↗
```

Nedster assigns work by role, sends only relevant project context, requires schema-shaped plans/actions, resolves every file beneath the selected project, and serializes mutating steps. It records failed approaches as temporary no-good branches, permits state-sensitive checks after a real edit, and prefers the smallest diagnostic-supported repair. Read-only work can run concurrently. This reduces wasted context and allows a smaller local model to behave more reliably than an unconstrained larger model.

For software tasks, completion is evidence-based:

1. inspect the actual project and select the smallest useful action;
2. edit files instead of returning placeholder code;
3. use compiler and test failures to revise the implementation;
4. rerun meaningful tests after the latest mutation;
5. exercise user-facing programs with representative input and an expected-output assertion.

The loop is bounded, so persistence does not mean uncontrolled token use or repeating an unchanged failure.

The optional `rag` dependency group provides ChromaDB, sentence-transformers, and BM25 building blocks:

```bash
python -m pip install -e ".[rag]"
```

## What works now

- Hardware probing and Ollama model discovery
- Guided and fully non-interactive setup (`--yes`, `--no-input`, `--dry-run`)
- Structured router/worker/critic loop with Ollama
- Project-scoped read, semantic search, write, patch, diff, and approved command tools
- Automatic Cargo, Go, Node.js, and Python test-runner selection
- Post-change tests and expected-output smoke checks for runnable applications and CLIs
- Public-web search/fetch fallback for current information such as market data
- Evidence-grounded answers that distinguish confirmed results from insufficient context
- Capability-bound tool requests, policy checks, sandbox selection, and tamper-evident local events
- Quiet terminal UI with optional `F2` execution details, plus a one-shot CLI
- Optional, explicitly configured OpenAI reasoning adapter
- Legacy MCP stdio server foundations while the unified MCP client/server manager is completed

The alpha does **not** yet match every feature in Cursor, Claude Code, Codex, or other mature cloud agents. IDE-grade inline completion, universal MCP lifecycle management, multimodal model routing, checkpoints/undo, and polished app bridges are on the roadmap. Contributions in these areas are especially welcome.

## Configuration and privacy

Generated state lives under `~/.nedster/`:

```text
~/.nedster/
├── config.toml
├── models/modelfiles/
└── runtime/
```

Local-only is the default privacy boundary. Nedster does not need an account, does not enable cloud inference automatically, and binds Ollama to its configured local endpoint. Files can still contain malicious instructions, so tool output and retrieved content are treated as untrusted. See [SECURITY.md](SECURITY.md) for the practical security model.

## Development

```bash
git clone https://github.com/unrealumanga/Nedster.git
cd Nedster
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check nedster_agent ironclad tests
python -m nedster_agent setup --dry-run --no-input
```

Architecture details live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), with upcoming work in [docs/ROADMAP_2026.md](docs/ROADMAP_2026.md).

## Who this is for

Nedster is for developers and makers looking for a **local AI coding agent**, **Ollama coding agent**, **autonomous coding assistant**, **private coding assistant**, **self-hosted AI assistant**, **local LLM agent**, **project-aware RAG agent**, **terminal AI agent**, or **MCP-ready automation agent** that can grow beyond coding into desktop and creative automation.

MIT licensed. Issues and contributions are welcome.
