<div align="center">

```text
 ██   ██  ███████  ██████   ███████  ███████  ███████  ██████  
 ███  ██  ██       ██   ██  ██         ███    ██       ██   ██ 
 ██ █ ██  █████    ██   ██  ███████    ███    █████    ██████  
 ██  ███  ██       ██   ██       ██    ███    ██       ██  ██  
 ██   ██  ███████  ██████   ███████    ███    ███████  ██   ██ 
                      Unchained Local AI
```

**An unstoppable, fully local, open-source coding agent that runs on your GPU.**

[![Local AI](https://img.shields.io/badge/AI-100%25%20Local-00E676?style=flat-square)](#)
[![VRAM](https://img.shields.io/badge/VRAM-8GB%2B-6200EA?style=flat-square)](#)
[![Context](https://img.shields.io/badge/Context-256K-FF5722?style=flat-square)](#)
[![TurboQuant](https://img.shields.io/badge/KV_Compression-TurboQuant-3F51B5?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](#)

</div>

Nedster is a highly autonomous, CLI-based AI software engineer designed to run entirely locally (tested and optimized for consumer hardware like the RTX 3060 Ti with 8GB VRAM). Powered by a heavily customized Ollama model (`aria-qwen`) and augmented with **Google's TurboQuant 4-bit KV Cache Compression**, it doesn't just chat—it *acts*. It searches your codebase, edits files precisely, scaffolds entire projects, and manages its own memory without ever sending a single line of your code to the cloud.

With TurboQuant integration, Nedster breaks the local VRAM barrier, allowing for a **massive 256K Context Window** on an 8GB GPU. You can feed it huge log files, entire codebases, and extensive documentation via RAG without triggering out-of-memory errors.

Inspired by premium agents like Devin and Claude Code, but built for the privacy-conscious developer who wants maximum autonomy on their own hardware.

---

## 🔥 Core Superpowers

- **Anti-Execution Theater:** Nedster is hard-wired to verify its own actions. If a command fails, it doesn't lie or hallucinate success. It reads the exit code, realizes the file wasn't created, and tries a different approach.
- **OpenClaw-Class Autonomy:** Runs up to 15 execution loops per prompt. Ask it to "refactor the auth module" and watch it plan, grep, read, edit, and verify sequentially without stopping.
- **Plan-Act-Verify (PAV) Engine:** For complex tasks ("build a microservice"), Nedster automatically generates a 3-5 step plan using a structured JSON task list (`todowrite`), executes each tool call, and physically verifies the files exist on disk before marking the task complete.
- **Dynamic Skill System:** Nedster learns. If it encounters a task it doesn't know, it can fetch docs, figure it out, and save the workflow to `~/.agents/skills/`. On every boot, it loads these learned capabilities directly into its context.
- **Context Preservation:** Rolls short-term context dynamically to save VRAM and saves session milestones into a persistent local ChromaDB. At 80% context usage, it automatically triggers an emergency compression to prevent "Tool Amnesia."

## 🛠️ Precision Tooling

Nedster comes equipped with a massive arsenal of deterministic tools, bypassing brittle XML code-diffing:
- `glob_search` / `grep_search`: Blazing fast regex and pattern matching across your project (uses `ripgrep` if installed).
- `edit_file`: Exact string replacement—guarantees safe, localized code updates.
- `scaffold_project`: Builds complete, multi-file project structures (including `git init` and `.venv`) atomically in one shot.
- `web_fetch`: Ingests web documentation seamlessly into context.

---

## 🤖 The Sidekicks Ecosystem

Nedster's true power lies in its adaptability. It comes bundled with two powerful "Sidekick" applications demonstrating how it can natively control external systems:

*   **ClawBrowser (`sidekicks/clawbrowser/`):** An Electron-based programmable headless browser. Tell Nedster to interact with a website, and it will write custom JavaScript payloads and inject them directly into ClawBrowser to automate web tasks, bypassing CAPTCHAs and React Shadow DOMs.
*   **H2Wealth (`sidekicks/h2wealth/`):** A high-performance, automated crypto trading bot operating on Bybit. This showcases Nedster's ability to understand complex architectures. You can ask Nedster to "Increase my leverage to 10x" or "Run a PNL analysis," and it will dynamically edit the `.env` configs, read the SQLite databases, and execute the analysis scripts.

---

## ⚡ Quickstart

### Prerequisites
- Windows, Linux, or macOS
- [Ollama](https://ollama.com/) installed
- Python 3.10+
- 8GB+ VRAM recommended

### 🐧 Linux / macOS Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/unrealumanga/Nedster.git
   cd Nedster
   ```

2. **Initialize the environment & build the model:**
   Run the startup script. This will download the base `qwen3.5:9b` weights, bake in the custom Nedster Modelfile, and set up the optimized execution environment (Flash Attention 2, Q8_0 KV Cache).
   ```bash
   ./start.sh
   ```

3. **Set up the Python Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Launch Nedster:**
   ```bash
   python main.py chat
   ```

### 🪟 Windows Setup
1. Download Ollama for Windows and ensure Python 3.10+ and Git are installed.
2. Clone the repository and open the folder.
3. Double-click `setup.bat` (this builds the venv, installs dependencies, and compiles the custom `aria-qwen` model).
4. Run `start.bat` for standard execution, OR run **`start_turboquant.bat`** to boot Nedster with the 256K Context Window and TurboQuant Server enabled!

---

## 💻 Usage & Commands

Inside the Nedster REPL, you can talk to it naturally:
> `Nedster> build a simple python hello world project at /tmp/hello_world`
> `Nedster> /pav implement a fastapi backend with sqlite in ./server`

**Slash Commands:**
- `/pav <task>` : Force the Plan-Act-Verify workflow for complex tasks.
- `/project <path>` : Switch Nedster's active workspace directory.
- `/clear` : Clear short-term memory (helps if context gets confused).
- `/fresh` : Nuclear reset. Completely reinitializes the agent to a blank slate.
- `/stats` : View real-time VRAM usage, token budget, active tasks, and vector counts.
- `/quicktest` : Run a rapid self-diagnostic of all core filesystem tools.
- `/tools` : List all active capabilities.

---

### 📦 Where are the model weights?

This repository contains the **core Nedster agent application code** (the orchestrator, RAG system, tools, and UI). It does **not** contain the raw `.gguf` weight files, as they are >6 GB in size. When you run the setup scripts, Nedster automatically triggers Ollama to pull the required base weights from its registry and dynamically compiles them using the `Modelfile` to create the specialized `aria-qwen` agent locally on your machine.

---

<div align="center">
  <b>#Nedster #TurboQuant #RAG #LocalLLM #Ollama #Qwen #256KContext #KVCacheCompression #AI #CodingAgent</b>
</div>
