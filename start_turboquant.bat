@echo off
setlocal enabledelayedexpansion

echo === Nedster with TurboQuant Backend ===
echo.
echo TurboQuant compresses KV Cache to 4-bits.
echo This allows for massive context windows (16K+) on 8GB VRAM.
echo.

:: Check Virtual Environment
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: venv not found. Please run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

:: Install required turboquant & openai if not present
python -c "import turboquant" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing TurboQuant and OpenAI...
    pip install turboquant[server] openai
)

:: Model selection
set MODEL=DavidAU/Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT
echo Using Model: %MODEL%

:: Start TurboQuant Server
echo.
echo Starting TurboQuant Inference Server on port 8000...
echo Keep this window open!
start /B turboquant-server --model %MODEL% --bits 4 --port 8000

echo Waiting for server to spin up...
timeout /T 15 /NOBREAK >nul

:: Set Env Vars for Nedster
set USE_TURBOQUANT=1
set TURBOQUANT_CONTEXT_SIZE=262144

echo === Nedster is ready with TurboQuant ===
echo Launching nedster...

python nedster.py

pause
