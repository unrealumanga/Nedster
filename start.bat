@echo off
where nedster >nul 2>nul
if errorlevel 1 (
  echo Nedster is not installed globally. Run: powershell -ExecutionPolicy Bypass -File install.ps1
  exit /b 1
)
nedster --project "%CD%" %*
