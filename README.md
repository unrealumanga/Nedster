# 🐺 Nedster

**An unstoppable, fully local, open-source coding agent that runs on your GPU.**

Nedster is a highly autonomous, CLI-based AI software engineer designed to run entirely locally (tested and optimized for an RTX 3060 Ti 8GB VRAM). Powered by a heavily customized Ollama model (`aria-qwen`), it doesn't just chat—it *acts*. It searches your codebase, edits files exactly, scaffolds entire projects, and manages its own memory without ever sending a single line of your code to the cloud.

Inspired by OpenClaw and Claude Code, but built for the privacy-conscious developer who wants maximum autonomy on consumer hardware.

---

## 🔥 Superpowers

- **100% Local & Private:** No API keys, no subscriptions, no telemetry. Your code stays on your machine.
- **Anti-Execution-Theater:** Nedster is hard-wired to verify its own actions. If a command fails, it doesn't lie or hallucinate success; it reads the exit code, realizes it failed, and tries again.
- **Deep Autonomy:** Runs up to 15 execution loops per prompt. Ask it to "refactor the auth module" and watch it plan, grep, read, edit, and verify sequentially without stopping.
- **Long-Term Memory:** Rolls short-term context dynamically to save VRAM and saves session milestones into a persistent local ChromaDB. It remembers what you discussed yesterday.

### OpenClaw-Style Precision Tools

Nedster comes equipped with a massive arsenal of deterministic tools, completely bypassing brittle XML code-diffing:
- `glob_search` / `grep_search`: Blazing fast regex and pattern matching across your project (uses `ripgrep` if installed).
- `edit_file`: Exact string replacement—guarantees safe, localized code updates.
- `scaffold_project`: Builds complete, multi-file project structures (including `git init` and `venv`) atomically in one shot.
- `todowrite`: Maintains a structured JSON task list to track complex multi-step goals.
- `web_fetch`: Ingests web documentation seamlessly into context.

---

## ⚡ Quickstart

### Prerequisites
- Linux / macOS
- [Ollama](https://ollama.com/) installed
- Python 3.10+
- 8GB+ VRAM (optimized for RTX 3060 Ti / RTX 4060)

### Installation

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

---

## 💻 Usage & Commands

Inside the Nedster REPL, you can talk to it naturally:
> `Nedster> build a simple python hello world project at /tmp/hello_world`

**Slash Commands:**
- `/project <path>` : Switch Nedster's active workspace directory.
- `/clear` / `/fresh` : Instantly dump short-term memory and restore clean tool access.
- `/stats` : View real-time VRAM usage, token budget, and vector counts.
- `/quicktest` : Run a rapid self-diagnostic of all core filesystem tools.
- `/tools` : List all active capabilities.

---

## 🧠 Architecture Notes

Nedster solves the "Tool Amnesia" problem inherent in mid-size LLMs (where they forget they have tools when the context window fills up). 
- It features an **Emergency Context Reset**: At 85% context usage, it automatically summarizes the session and flushes raw messages to preserve tool instructions.
- It uses a **Direct Execution Router** for simple file operations (`create file X`, `read Y`), bypassing the LLM entirely for zero-latency operations.
- It dynamically strips "I am an AI" refusal language from its own memory stream.

Enjoy your new local, unstoppable AI engineer.
