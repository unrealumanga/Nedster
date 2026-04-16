@echo off
setlocal enabledelayedexpansion

REM ── Install mode: adds nedster to PATH ────────────────────
if "%1"=="--install" goto :install
if "%1"=="-i"        goto :install
goto :run

:install
echo Installing nedster to PATH...
set "NEDSTER_DIR=%~dp0"
REM Remove trailing backslash
set "NEDSTER_DIR=%NEDSTER_DIR:~0,-1%"

REM Create nedster.bat in a permanent location
set "BIN_DIR=%USERPROFILE%\AppData\Local\nedster-bin"
if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"

REM Write the launcher
(
  echo @echo off
  echo cd /d "%NEDSTER_DIR%"
  echo call venv\Scripts\activate.bat
  echo python nedster.py %%*
) > "%BIN_DIR%\nedster.bat"

REM Add to user PATH permanently
for /f "tokens=2*" %%A in ('reg query HKCU\Environment /v PATH 2^>nul') do set "CURPATH=%%B"

echo "%CURPATH%" | findstr /i "%BIN_DIR%" >nul
if errorlevel 1 (
    reg add HKCU\Environment /v PATH /d "%CURPATH%;%BIN_DIR%" /f >nul
    echo Added to PATH: %BIN_DIR%
    echo.
    echo IMPORTANT: Open a NEW terminal for 'nedster' to work.
    echo In new terminal: nedster
) else (
    echo Already in PATH.
)
goto :eof

:run
REM ── Normal start ───────────────────────────────────────────
echo === Aria RAG Stack - Starting (Windows) ===

:: Speed env vars for Ollama
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_NUM_THREADS=8
set OMP_NUM_THREADS=8

:: Build the custom Aria model from Modelfile (only if not already built)
ollama list | findstr /I "qwen3.5:9b" >nul
if %ERRORLEVEL% neq 0 (
    echo Pulling base model...
    ollama pull qwen3.5:9b
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
start /B ollama run qwen3.5:9b "ping" >nul 2>&1
echo Model warm.

:: Check Virtual Environment
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: venv not found. Please run setup.bat first.
    pause
    exit /b 1
)

echo === Aria is ready ===
echo Launching nedster...

:: Automatically start the REPL
call venv\Scripts\activate.bat
python nedster.py

pause
