#!/bin/bash
# File: start.sh
set -e
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_NUM_THREADS=8
export OLLAMA_KEEP_ALIVE=15m
export OLLAMA_MAX_LOADED_MODELS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect python binary
PYTHON=$(which python3 2>/dev/null || which python 2>/dev/null)
if [ -z "$PYTHON" ]; then echo "ERROR: python not found"; exit 1; fi

# Activate existing venv
if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
    echo "Venv activated: $VIRTUAL_ENV"

# Ensure num_gpu is removed so Ollama auto-allocates memory safely
sed -i "/^PARAMETER num_gpu/d" ./Modelfile
echo "VRAM strict allocation removed. Ollama will auto-manage memory."

# Rebuild model without strict num_gpu
ollama create aria-qwen -f ./Modelfile 2>/dev/null && echo "Model rebuilt OK"


else
    echo "ERROR: venv not found. Run: python3 -m venv ./venv && pip install -r requirements.txt"
    exit 1
fi

# Start Ollama if not running
if ! pgrep -x ollama > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 3
fi

# Build aria-qwen model if not exists
if ! ollama list 2>/dev/null | grep -q "aria-qwen"; then
    echo "Building aria-qwen from Modelfile..."
    ollama create aria-qwen -f ./Modelfile
fi

# Async warm-up: don't block startup
(ollama run aria-qwen "" > /dev/null 2>&1 || true) &
WARMUP_PID=$!
echo "Model warming in background (PID: $WARMUP_PID)..." 

# Background heartbeat: prevent VRAM unload (ping every 240s)
(while true; do
    sleep 200
    ollama run aria-qwen "" > /dev/null 2>&1 || true
done) &
HEARTBEAT_PID=$!
trap "kill $HEARTBEAT_PID 2>/dev/null; echo 'Heartbeat stopped'" EXIT

# Verify imports
python3 -c "
import rank_bm25, chromadb, sentence_transformers, tiktoken, fitz, torch, tqdm, psutil, ollama
print('All imports OK')
" || { echo "ERROR: Missing packages. Run: pip install -r requirements.txt"; exit 1; }

# Create required dirs
mkdir -p ~/.aria
mkdir -p ~/AI_Lab/archives
mkdir -p ~/AI_Lab/quarantine
touch ~/.aria/milestones.md
touch ~/.aria/workflows.jsonl

echo -e "\033[1;36m"
echo -e "\033[1;36m ██   ██  ███████  ██████   ███████  ███████  ███████  ██████  \033[0m"
echo -e "\033[1;36m ███  ██  ██       ██   ██  ██         ███    ██       ██   ██ \033[0m"
echo -e "\033[1;36m ██ █ ██  █████    ██   ██  ███████    ███    █████    ██████  \033[0m"
echo -e "\033[1;36m ██  ███  ██       ██   ██       ██    ███    ██       ██  ██  \033[0m"
echo -e "\033[1;36m ██   ██  ███████  ██████   ███████    ███    ███████  ██   ██ \033[0m"
echo -e "                      \033[38;5;245mUnchained Local AI\033[0m"
echo ""
echo "=== Aria RAG Stack Ready ==="
echo "Flash Attention: $OLLAMA_FLASH_ATTENTION"
echo "KV Cache: $OLLAMA_KV_CACHE_TYPE"
echo "Keep Alive: $OLLAMA_KEEP_ALIVE"
echo "Heartbeat PID: $HEARTBEAT_PID"
echo "Python: $(which python3)"
echo ""
echo "Starting chat..."

echo "==========================="
echo "  Nedster - Main Menu"
echo "==========================="
echo "1) Start Interactive TUI Agent"
echo "2) Start Web Dashboard"
echo "3) Replay a Session"
echo "4) Run a one-shot command"
echo "5) Run a swarm command"
echo ""
read -p "Choose an option [1]: " choice
choice=${choice:-1}

case $choice in
    1)
        python3 nedster.py repl
        ;;
    2)
        ./start_web.sh
        ;;
    3)
        python3 nedster.py replay
        ;;
    4)
        read -p "Enter your one-shot command: " prompt
        python3 nedster.py oneshot "$prompt"
        ;;
    5)
        read -p "Enter your swarm command: " prompt
        python3 nedster.py swarm "$prompt"
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac
