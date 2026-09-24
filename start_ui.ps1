$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvDir    = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$activate   = Join-Path $venvDir "Scripts\Activate.ps1"

# 1. Crear la venv si no existe o está rota
if (-not (Test-Path $venvPython)) {
    Write-Host "Creando .venv..."
    if (Test-Path $venvDir) { Remove-Item -Recurse -Force $venvDir }
    $py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
    & $py -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { throw "No se pudo crear la virtualenv." }
}

. $activate

# 2. Instalar dependencias solo si cambió requirements-api.txt
$req   = Join-Path $projectRoot "requirements-api.txt"
$stamp = Join-Path $venvDir ".requirements.hash"
$hash  = (Get-FileHash $req -Algorithm SHA256).Hash
if (-not (Test-Path $stamp) -or (Get-Content $stamp -Raw).Trim() -ne $hash) {
    Write-Host "Instalando dependencias..."
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r $req
    if ($LASTEXITCODE -ne 0) { throw "Falló la instalación de dependencias." }
    Set-Content $stamp $hash
}

$env:PYTHONPATH = $projectRoot

# 3. Lanzar la API en otra ventana
$apiCmd = "Set-Location '$projectRoot'; . '$activate'; `$env:PYTHONPATH='$projectRoot'; `$env:VLLM_BASE_URL='http://localhost:11434/v1'; `$env:MODEL_SERVE_NAME='celtia-qwen3'; `$env:MODEL_CONTEXT='8192'; `$env:GATEWAY_MAX_CONCURRENCY='4'; `$env:PUBLIC_BASE_URL='https://celtiaia.com'; python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8081 --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $apiCmd
Write-Host "API en http://localhost:8081"

# 4. Abrir el navegador y servir la web
Start-Job { Start-Sleep 2; Start-Process "http://localhost:4173" } | Out-Null
Write-Host "Demo web en http://localhost:4173 (Ctrl+C para parar)"
# 5. Estado del túnel público (celtiaia.com). Si falla: servicio "Cloudflared" caído o token caducado
Start-Job {
    Start-Sleep 20
    try { $r = Invoke-WebRequest "https://celtiaia.com/health" -UseBasicParsing -TimeoutSec 15; Write-Host "celtiaia.com -> $($r.StatusCode)" }
    catch { Write-Host "celtiaia.com NO responde. Revisa el servicio Cloudflared (scripts\install-cloudflared-service.ps1, como administrador)." }
} | Out-Null

python -m http.server 4173 --bind 127.0.0.1 --directory apps/web
