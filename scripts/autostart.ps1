# Arranca API (8081) e web demo (4173) sen ventás, sen --reload. Úsao a tarefa programada "CeltIA-Autostart".
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:PYTHONPATH = $root
$env:VLLM_BASE_URL = "http://localhost:11434/v1"
$env:MODEL_SERVE_NAME = "celtia-qwen3"
$env:MODEL_CONTEXT = "8192"
$env:GATEWAY_MAX_CONCURRENCY = "4"
$env:PUBLIC_BASE_URL = "https://celtiaia.com"
# Primeiro paso do rollout CDE: shadow por defecto, pero respecta unha promoción/rollback explícitos.
if ([string]::IsNullOrWhiteSpace($env:DECISION_ROUTING_MODE)) {
    $env:DECISION_ROUTING_MODE = "shadow"
}
if ([string]::IsNullOrWhiteSpace($env:DECISION_CDE_ROLLOUT_PERCENT)) {
    $env:DECISION_CDE_ROLLOUT_PERCENT = "0"
}

function Test-Port($p) { [bool](Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue) }

if (-not (Test-Port 8081)) {
    Start-Process $py -ArgumentList "-m","uvicorn","apps.api.main:app","--host","127.0.0.1","--port","8081" -WorkingDirectory $root -WindowStyle Hidden
}
if (-not (Test-Port 4173)) {
    Start-Process $py -ArgumentList "-m","http.server","4173","--bind","127.0.0.1","--directory","apps/web" -WorkingDirectory $root -WindowStyle Hidden
}
