@echo off
setlocal enabledelayedexpansion

echo === Nedster Windows Setup ===
echo Please ensure you have Python 3.10+ and Git installed.

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in PATH. Please install Python 3.10+
    exit /b 1
)

:: Create Virtual Environment
if not exist "venv\Scripts\activate.bat" (
    echo [1/3] Creating Python Virtual Environment...
    python -m venv venv
) else (
    echo [1/3] Virtual environment already exists.
)

:: Activate and install dependencies
echo [2/3] Installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt

:: Check Ollama
ollama --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [3/3] Ollama not found. Please install Ollama for Windows from https://ollama.com/download/windows
) else (
    echo [3/3] Pulling base model...
    ollama pull qwen3.5:9b
    echo Building aria-qwen model...
    ollama create aria-qwen -f Modelfile
)

echo.
echo Setup Complete! 
echo To run Nedster, double-click start.bat or type "start.bat" in the terminal.
pause
