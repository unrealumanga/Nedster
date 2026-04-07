#!/bin/bash
# Install Ollama on Linux:
curl -fsSL https://ollama.com/install.sh | sh
# Pull model with Q4_K_M quantization:
ollama pull qwen3.5:9b
# Confirm GPU is being used:
ollama run qwen3.5:9b "test" && nvidia-smi
# Install Python deps:
pip install -r requirements.txt