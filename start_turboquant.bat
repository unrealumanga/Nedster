@echo off
setlocal enabledelayedexpansion

echo === Nedster with Local Ollama Backend ===
echo.
echo Bypassing TurboQuant Server. 
echo Utilizing the local Ollama instance directly.
echo.

:: Check Virtual Environment
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: venv not found. Please run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

:: Make sure Ollama is running
tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul
if %ERRORLEVEL% neq 0 (
    echo Starting Ollama server...
    start /B ollama serve
    timeout /T 3 /NOBREAK >nul
)

:: Model selection
if "%~1"=="" (
    set MODEL=aria-lfm2
) else (
    set MODEL=%~1
)
echo Using Local Ollama Model: !MODEL!

:: Set Env Vars for Nedster to bypass TurboQuant but keep large context
set USE_TURBOQUANT=0
set TURBOQUANT_CONTEXT_SIZE=262144
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q4_0

echo === Nedster is ready ===
echo Launching nedster...

python nedster.py

pause
