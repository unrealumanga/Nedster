# 👾 Nedster CLI Coding Agent

<div align="center">
  <p>
    <a href="#"><img src="https://img.shields.io/badge/AI-100%25%20Local-00E676?style=flat-square" alt="100% Local"></a>
    <a href="#"><img src="https://img.shields.io/badge/VRAM-8GB%2B-6200EA?style=flat-square" alt="8GB VRAM"></a>
    <a href="#"><img src="https://img.shields.io/badge/Context-256K-FF5722?style=flat-square" alt="256K Context"></a>
    <a href="#"><img src="https://img.shields.io/badge/KV_Compression-TurboQuant-3F51B5?style=flat-square" alt="TurboQuant"></a>
  </p>
</div>

**An unstoppable, fully local, open-source coding agent that runs on your consumer GPU.**

**Tags:** `ollama` `coding-agent` `local-ai` `cli` `rag` `chromadb` `python` `qwen`

Are you trying to use local LLMs to autonomously write code, read files, and manage your projects, only to watch them suffer from "tool amnesia," hallucinate XML tags, or get stuck in infinite execution loops? 

**Nedster** solves this instantly.

Nedster is a highly autonomous, CLI-based AI software engineer designed for privacy-conscious developers. Powered by Ollama and augmented with **TurboQuant 4-bit KV Cache Compression**, it doesn't just chat—it *acts*. It searches your codebase, edits files precisely, and scaffolds entire projects without ever sending a single line of your code to the cloud.

## ✨ Features
*   **🛡️ Bulletproof Tool Execution:** A highly fortified, single-pass regex parser catches broken XML, malformed JSON, and markdown fallbacks. If the model meant to run a tool, Nedster executes it.
*   **🧠 Amnesia Correction:** If the LLM forgets it has filesystem access and apologizes, Nedster dynamically intercepts the refusal, injects a system correction, and forces a retry.
*   **🗜️ 256K Context on 8GB VRAM:** Integrated with Google's TurboQuant. Feed Nedster massive log files and entire codebases without triggering out-of-memory errors.
*   **📚 Built-in Local RAG:** Comes with a standalone ChromaDB engine to vectorize and semantically search your project directories effortlessly.
*   **🔁 Iteration Budgets:** Hard limits on autonomous loops and continuity watchdogs ensure the model stays on track and never hangs your terminal.

## 🛠️ Quick Start

It takes just a few commands to get your local coding agent up and running.

**Windows:**
```bat
git clone https://github.com/unrealumanga/Nedster.git
cd Nedster
setup.bat
start.bat
```

**Linux / Mac:**
```bash
git clone https://github.com/unrealumanga/Nedster.git
cd Nedster
chmod +x setup.sh start.sh
./setup.sh
./start.sh
```

### Navigating the CLI
Once running, just type your request naturally:
```text
> fix the auth logic in src/auth.py to use the requests library
> scaffold a new React project in the frontend/ folder
> /stats
```

## 📈 System Monitoring
Nedster includes a beautiful Terminal UI (TUI) that provides real-time tracking of your CPU RAM, ChromaDB vectors, and GPU VRAM polling (via `nvidia-smi`), keeping you in total control of your hardware.