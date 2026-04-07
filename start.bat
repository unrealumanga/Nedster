@echo off
setlocal enabledelayedexpansion

echo === Aria RAG Stack - Starting (Windows) ===

:: Speed env vars for Ollama
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_NUM_THREADS=8
set OMP_NUM_THREADS=8

:: Build the custom Aria model from Modelfile (only if not already built)
ollama list | findstr /I "aria-qwen" >nul
if %ERRORLEVEL% neq 0 (
    echo Building Aria personality model...
    ollama create aria-qwen -f Modelfile
    echo Aria model created.
)

:: Start Ollama server in background if not running
tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul
if %ERRORLEVEL% neq 0 (
    echo Starting Ollama server...
    start /B ollama serve
    timeout /T 3 /NOBREAK >nul
)

:: Warm up the model
echo Warming up model...
start /B ollama run aria-qwen "ping" >nul 2>&1
echo Model warm.

:: Check Virtual Environment
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: venv not found. Please run setup.bat first.
    pause
    exit /b 1
)

echo === Aria is ready ===
echo Run: "call venv\Scripts\activate.bat" and "python main.py chat"

:: Automatically start the REPL
call venv\Scripts\activate.bat
python main.py chat

pause
