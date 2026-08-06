[CmdletBinding()]
param(
    [switch]$SkipSetup
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Installing the global Nedster command..." -ForegroundColor Cyan
if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv tool install --force $ProjectRoot
    $ScriptsPath = (& uv tool dir --bin).Trim()
    & uv tool update-shell | Out-Null
} else {
    $Python = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
    & $Python -m pip install --user --upgrade $ProjectRoot
    $ScriptsPath = (& $Python -c "import sysconfig; print(sysconfig.get_path('scripts', scheme='nt_user'))").Trim()
}

$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
$PathParts = @($UserPath -split ";" | Where-Object { $_ })
if ($PathParts -notcontains $ScriptsPath) {
    [Environment]::SetEnvironmentVariable("Path", (($PathParts + $ScriptsPath) -join ";"), "User")
}
if (($env:Path -split ";") -notcontains $ScriptsPath) {
    $env:Path = "$env:Path;$ScriptsPath"
}

$LauncherPath = Join-Path $ScriptsPath "nedster.exe"
if (-not (Test-Path -LiteralPath $LauncherPath)) {
    throw "Nedster installed, but the launcher is not visible. Open a new terminal and run: nedster setup"
}

Write-Host "Global launcher ready: $LauncherPath" -ForegroundColor Green
if (-not $SkipSetup) {
    & $LauncherPath setup
}
