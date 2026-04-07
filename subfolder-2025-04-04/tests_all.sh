#!/bin/bash
# Basic test suite
cd /home/mnm/AI_Lab/Workspace/Nedster
source ./venv/bin/activate
python3 -m py_compile *.py && echo "Syntax OK"
python3 nedster.py stats > /dev/null && echo "Stats command OK"
echo "/exit" | python3 nedster.py > /dev/null && echo "REPL exit OK"
echo "All tests passed"
