# Starts ResearchLens on Windows without Docker: Redis (inside WSL), the API,
# a Celery worker and the frontend, each in its own window, then opens the
# app in your browser once the API answers. The Docker equivalent is
# `docker compose up --build`.
#
#   .\start-dev.cmd            everything (needs Redis installed in WSL)
#   .\start-dev.cmd -Simple    no Redis or worker: reports run inside the API
#
# Stop a part by closing its window (or Ctrl+C in it).

param([switch]$Simple)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"
$activate = Join-Path $backend "venv\Scripts\Activate.ps1"

if (-not (Test-Path $activate)) {
    Write-Host "No backend\venv found. Create it first:" -ForegroundColor Red
    Write-Host "  cd backend; python -m venv venv; venv\Scripts\activate; pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path (Join-Path $backend ".env"))) {
    Write-Host "No backend\.env found. Copy backend\.env.example to backend\.env and fill in GROQ_API_KEY and JWT_SECRET_KEY." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "Installing frontend packages (first run only)..."
    Push-Location $frontend
    npm install
    Pop-Location
}

function Start-Window([string]$title, [string]$dir, [string]$command) {
    # -NoExit keeps the window open, so its output (and any error) stays readable.
    Start-Process powershell -WorkingDirectory $dir -ArgumentList @(
        "-NoExit", "-ExecutionPolicy", "Bypass",
        "-Command", "`$Host.UI.RawUI.WindowTitle = '$title'; $command"
    )
}

$venv = ". '$activate'"
if ($Simple) {
    Start-Window "ResearchLens API (simple mode)" $backend "$venv; `$env:CELERY_TASK_ALWAYS_EAGER = 'true'; uvicorn app.main:app --reload"
} else {
    Start-Window "ResearchLens Redis" $root "wsl -e redis-server"
    Start-Window "ResearchLens API" $backend "$venv; uvicorn app.main:app --reload"
    Start-Window "ResearchLens worker" $backend "$venv; celery -A app.celery_app worker --loglevel=info"
}
Start-Window "ResearchLens frontend" $frontend "npm run dev"

Write-Host "Waiting for the API to start (loading models can take a minute)..."
$deadline = (Get-Date).AddSeconds(120)
$up = $false
while ((Get-Date) -lt $deadline) {
    try {
        Invoke-WebRequest "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 2 | Out-Null
        $up = $true
        break
    } catch {
        Start-Sleep -Seconds 2
    }
}

if ($up) {
    Write-Host "ResearchLens is running: http://localhost:5173  (API docs: http://127.0.0.1:8000/docs)" -ForegroundColor Green
    Start-Process "http://localhost:5173"
} else {
    Write-Host "The API didn't answer within 2 minutes -- check the 'ResearchLens API' window for an error." -ForegroundColor Yellow
}
