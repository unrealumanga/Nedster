#!/bin/bash
# Launch script for Nedster

# Suppress HF warnings
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false

# Get the directory of this script
DIR="/home/mnm/AI_Lab/Workspace/Nedster"
cd "$DIR"

# Activate virtual environment
if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
else
    echo "Error: Virtual environment not found at $DIR/venv"
    exit 1
fi

# Pass all arguments to nedster.py
exec python3 nedster.py "$@"
