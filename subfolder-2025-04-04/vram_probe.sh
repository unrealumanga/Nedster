#!/bin/bash
# Probe available VRAM and calculate safe num_gpu for Qwen3.5-9B Q4_K_M
# Q4_K_M = ~5.5GB weights. Each layer = ~5.5GB/35 ≈ 157MB

VRAM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -z "$VRAM_MB" ]; then
    echo "0"
    exit 0
fi

# Reserve 1800MB for KV cache + CUDA overhead + Ollama buffers
USABLE=$((VRAM_MB - 1800))
if [ "$USABLE" -lt 0 ]; then
    echo "0"
    exit 0
fi

# Each layer ~157MB for Q4_K_M 9B model
LAYERS=$((USABLE / 157))
# Cap at 35 (total layers in 9B model)
if [ "$LAYERS" -gt 35 ]; then LAYERS=35; fi
# Floor at 0
if [ "$LAYERS" -lt 0 ]; then LAYERS=0; fi

echo "$LAYERS"
