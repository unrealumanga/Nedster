# 🔥 Nedster — Mojo Language Port

**High-Frequency Crypto Analytics, Local GGUF Agent Harness, Bare-Metal SIMD Execution & Terminal UI in Mojo (`.mojo`)**

Nedster has been converted to [Mojo](https://www.modular.com/mojo) to capitalize on bare-metal MLIR compilation, SIMD vectorization, zero-overhead Python interop (`ccxt`), and GIL-free concurrency for high-frequency crypto trading and local GGUF model execution.

---

## 🚀 Architecture Overview

```text
Nedster Mojo Architecture
│
├── nedster_mojo/hardware.mojo       # Probes CPU, RAM, NVIDIA CUDA, AMD ROCm, Apple Silicon Unified Memory
├── nedster_mojo/inference.mojo      # SIMD GGUF Quantization Math (Q4_0, Q8_0) & Ollama REST / OpenAI API
├── nedster_mojo/chart_analytics.mojo# Bare-metal SIMD MACD, RSI, OFI, CVD, Funding Z-score, ADX
├── nedster_mojo/exchange_api.mojo   # Zero-overhead Python CCXT interop & High-Frequency WebSocket Streams
├── nedster_mojo/sandbox.mojo        # Security Policy Engine & Process Isolation
├── nedster_mojo/tools.mojo          # Sandboxed File IO, Git Diff, Search, Shell Execution
├── nedster_mojo/rag_engine.mojo     # SIMD Vector Search & Evidence Grounding Harness
├── nedster_mojo/orchestrator.mojo   # Multi-role Router / Worker / Critic Agent Loops
├── nedster_mojo/tui.mojo            # Monospace Terminal UI
└── nedster_mojo/main.mojo           # CLI Entrypoint (`fn main() raises:`)
```

---

## ⚡ Why Mojo for Nedster Crypto & AI

1. **Bare-Metal SIMD Math:** Chart indicators (RSI, MACD, Order Flow Imbalance, Cumulative Volume Delta) execute using native 8-wide float32 SIMD registers (`SIMD[DType.float32, 8]`).
2. **Local GGUF Acceleration:** Native vector matrix operations for dequantizing Q4_0 / Q8_0 GGUF tensors without Python interpreter overhead.
3. **Zero-Overhead CCXT Integration:** Direct access to Python's industry-standard `ccxt` library via `from python import Python`, streaming orderbooks directly into Mojo structs.
4. **No GIL Bottleneck:** Full CPU core utilization for concurrent WebSocket exchange data streams and local LLM inference harness.

---

## 🛠️ Usage

### Running the Interactive TUI
```bash
mojo nedster_mojo/main.mojo
```

### One-Shot Command Line Execution
```bash
mojo nedster_mojo/main.mojo run "Analyze BTC/USDT market signals"
mojo nedster_mojo/main.mojo run "List project files"
```

### Hardware Diagnostic Doctor
```bash
mojo nedster_mojo/main.mojo doctor
```

---

## 📊 Comparative Performance Matrix

| Feature | Python | Rust | Go | C++ | 🔥 Mojo (Nedster) |
|---|---|---|---|---|---|
| **API Connectivity** | ✅ Easy (`ccxt`) | ⚠️ Fragmented | ⚠️ Fragmented | ❌ Hard | ✅ **Best** (Seamless Python `ccxt`) |
| **Signal Latency** | ❌ Slow (GIL) | ✅ Sub-ms | ⚠️ Fast (GC pauses) | ✅ Sub-ms | ✅ **Sub-ms** (Bare-metal MLIR) |
| **Local GGUF / AI** | ⚠️ Wrapper overhead | ✅ Fast | ❌ Poor | ✅ Fast | ✅ **Native SIMD & MLIR** |
| **Dev Speed** | ✅ Very Fast | ❌ Complex | ✅ Fast | ❌ Slow | ✅ **Fast (Python Syntax)** |
