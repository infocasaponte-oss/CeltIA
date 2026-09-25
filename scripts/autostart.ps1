param(
    [switch]$RestartApi
)

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

function Stop-CeltiaApiListener {
    $listeners = Get-NetTCPConnection -LocalPort 8081 -State Listen -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        $processId = [int]$listener.OwningProcess
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction SilentlyContinue
        $cmd = if ($proc) { [string]$proc.CommandLine } else { "" }
        if ($cmd -notmatch "uvicorn" -or $cmd -notmatch "apps\.api\.main:app") {
            throw "O porto 8081 está ocupado polo PID $processId, pero non parece ser a API de CeltIA. Non se matou ningún proceso."
        }
        Stop-Process -Id $processId -Force -ErrorAction Stop
    }
    for ($i = 0; $i -lt 40 -and (Test-Port 8081); $i++) {
        Start-Sleep -Milliseconds 250
    }
    if (Test-Port 8081) {
        throw "A API antiga non liberou o porto 8081."
    }
}

if ($RestartApi -and (Test-Port 8081)) {
    Stop-CeltiaApiListener
}

if (-not (Test-Port 8081)) {
    Start-Process $py -ArgumentList "-m","uvicorn","apps.api.main:app","--host","127.0.0.1","--port","8081" -WorkingDirectory $root -WindowStyle Hidden
}
if (-not (Test-Port 4173)) {
    Start-Process $py -ArgumentList "-m","http.server","4173","--bind","127.0.0.1","--directory","apps/web" -WorkingDirectory $root -WindowStyle Hidden
}
