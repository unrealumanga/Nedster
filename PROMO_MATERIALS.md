# Nedster Promotional Materials

Use these tailored drafts to promote Nedster across different developer communities. 

---

## 1. Hacker News (Show HN)
*Link: https://news.ycombinator.com/submit*
*Note: Hacker News loves technical details, origin stories, and a humble tone. Avoid overly marketing-speak; focus on the engineering problems Nedster solves.*

**Title:** 
Show HN: Nedster – A local, tool-capable CLI AI agent built on Ollama

**Body:**
Hi HN,

I’ve been building Nedster, an open-source interactive CLI agent designed to run locally with Ollama. 

Like many here, I love the idea of local LLMs managing my filesystem, running bash commands, and refactoring code. But I kept running into issues with smaller models (like Qwen or Llama variants): they suffer from "tool amnesia," hallucinate XML tags, fall back to raw markdown, or get stuck in infinite execution loops. 

To fix this, I built Nedster with a heavily fortified execution loop. Some of the core engineering:
*   **Robust Tool Parser:** A single-pass regex parser that catches broken XML, malformed JSON, and bare markdown code blocks, ensuring the model's intent actually translates to tool execution.
*   **Iteration Budgets & Continuity Watchdogs:** Hard caps on iterations (to prevent infinite loops) and a watchdog daemon to ensure the agent doesn't just go silent. 
*   **Self-Correction:** If the model forgets it has filesystem access, the agent intercepts the refusal, injects a system correction, and forces a retry.
*   **Resource Aware:** Built-in VRAM/RAM monitoring and local RAG via ChromaDB.

It includes tools for file reading/writing, AST-based Python syntax checking, web fetching, and robust grep/glob searching.

I’d love for you to try it out, poke holes in the architecture, and let me know what you think. 

Repo: https://github.com/unrealumanga/Nedster

---

## 2. Reddit
*Good Subreddits: r/LocalLLaMA, r/coding, r/Python, r/opensource*
*Note: The Reddit local AI community loves performance tracking, handling "dumb" models, and clear feature lists.*

**Title:** 
I got tired of local LLMs forgetting how to use tools, so I built Nedster: A robust CLI Agent for Ollama. 

**Body:**
Hey everyone! If you’ve ever tried to give a 7B or 9B parameter model access to your filesystem, you know the pain: they hallucinate tool tags, write malformed JSON, or just give up and output raw markdown blocks. 

I built **Nedster** to solve exactly this. It's a Python-based CLI agent that connects to Ollama, but with a massively reinforced tool-execution engine.

**Key Features:**
*   **Bulletproof Tool Parsing:** If your model spits out broken `<tool>` tags, missing slashes, or just defaults to ` ```bash `, Nedster intercepts it and executes it correctly.
*   **Amnesia Correction:** If the LLM says "I am an AI and cannot access your local files," Nedster intercepts the output, injects a system correction telling the model to use its tools, and retries automatically.
*   **Built-in Safety & Limits:** Uses an `IterationBudget` so the LLM doesn't loop forever.
*   **System Stats:** TUI displays real-time CPU/RAM usage and GPU VRAM tracking (via `nvidia-smi`).
*   **Local RAG:** Integrated ChromaDB vector store for project context.

It's completely open-source and meant for people who want local, private coding assistants that *actually* get things done. 

Check it out on GitHub: https://github.com/unrealumanga/Nedster
Would love to hear your feedback or feature requests!

---

## 3. Twitter / X (Thread)
*Note: Twitter needs a strong hook, visual emojis, and bite-sized architecture details. (Highly recommend attaching a screenshot or GIF of the CLI terminal to the first tweet!)*

**Tweet 1:**
Local LLMs are amazing, but getting them to reliably execute tools without hallucinating or breaking JSON is a nightmare. 

Meet **Nedster** 👾: A robust, fully-local CLI agent powered by Ollama that actually gets things done. 

Thread on how it works 👇🧵
#LocalLLM #Ollama #Python #AI

**Tweet 2:**
Problem 1: "Tool Amnesia" 🧠❌
Smaller models constantly forget they have filesystem access and apologize. 
Nedster detects this "amnesia", blocks the apology, and dynamically injects a system correction forcing the model to use its tools.

**Tweet 3:**
Problem 2: Broken Output 🛠️
Models output malformed XML, broken JSON, or just raw markdown. 
Nedster uses a heavily fortified single-pass regex parser that catches and repairs almost any broken tool-call format. If the model meant to do it, Nedster runs it.

**Tweet 4:**
Problem 3: Infinite Loops ♾️
Nedster implements an `IterationBudget` and a `ContinuityWatchdog`. It caps autonomous loops and ensures your terminal doesn't hang forever, keeping your context window safe and saving your compute.

**Tweet 5:**
Plus, it comes with a beautiful Terminal UI, real-time GPU VRAM tracking, and ChromaDB for local RAG context. 

If you want a private, local AI coding sidekick, try it out:
🔗 https://github.com/unrealumanga/Nedster
Star the repo if you like it! ⭐️

---

## 4. Dev.to / Medium / Hashnode
*Note: Blog platforms are great for "How I built this" narratives and deep-dives into the code.*

**Title:** Building a Local LLM Agent that Actually Works: Meet Nedster
**Subtitle:** How I fixed "tool amnesia", infinite loops, and malformed JSON in local AI agents.

**Body:**
Running AI agents locally is the dream: zero latency, complete privacy, and no API costs. But if you’ve ever hooked up a local 8B model to a coding workflow, you’ve likely hit a wall. They forget they have tools, output broken XML, or trap themselves in endless loops. I built **Nedster** to fix the bridge between local models and actual execution.

### The "Tool Amnesia" Problem
Often, smaller models will default to their safety training: *"I am an AI and cannot modify your files."* 
In Nedster, I implemented a detection layer. If the agent detects an apology or refusal to use the filesystem, it stops the output, injects a forceful system prompt reminding the model of its tools, and regenerates the response seamlessly. 

### Fortified Parsing
We can't expect local models to output perfect JSON 100% of the time. Nedster handles everything from malformed `<tool>` tags to fallback markdown bash blocks. The `ToolExecutor` cleans up the mess and runs the command. 

### Iteration Budgets & VRAM Tracking
Context windows aren't infinite. Nedster uses an `IterationBudget` that cuts the model off if it spins its wheels too long. I also added a Terminal UI (TUI) that polls `nvidia-smi` to give you live VRAM tracking—crucial when running models locally.

### Try it yourself
Nedster is open-source, runs on Python, and connects seamlessly to Ollama. Let's make local AI useful again. 

Check it out and leave a star: https://github.com/unrealumanga/Nedster