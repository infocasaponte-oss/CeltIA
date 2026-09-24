$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$activateScript = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    throw "No se encontró la virtualenv en .venv. Crea la venv antes de ejecutar este script."
}

. $activateScript

$env:PYTHONPATH = $projectRoot
Write-Host "Activada la venv de Mini-Council. Iniciando API..."
uvicorn apps.api.main:app --host 127.0.0.1 --port 8081 --reload
