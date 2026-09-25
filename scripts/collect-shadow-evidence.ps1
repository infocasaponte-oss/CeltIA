param(
    [string]$ApiBase = "http://127.0.0.1:8081",
    [string]$AdminToken,
    [int]$Samples = 220,
    [int]$DelayMs = 500,
    [string]$KeyName = "shadow-rollout",
    [int]$MaxTokens = 96
)

$ErrorActionPreference = "Stop"

function Get-AdminTokenValue {
    param([string]$Token)

    if (-not [string]::IsNullOrWhiteSpace($Token)) { return $Token }
    if (-not [string]::IsNullOrWhiteSpace($env:ADMIN_TOKEN)) { return $env:ADMIN_TOKEN }

    $candidates = @(
        (Join-Path $PSScriptRoot "..\.env"),
        (Join-Path $PSScriptRoot "..\.env.local")
    )

    foreach ($path in $candidates) {
        if (-not (Test-Path $path)) { continue }
        foreach ($line in Get-Content $path) {
            if ($line -match '^\s*#' -or [string]::IsNullOrWhiteSpace($line)) { continue }
            if ($line -match '^\s*ADMIN_TOKEN\s*=\s*(.+?)\s*$') {
                return $Matches[1].Trim('"').Trim("'")
            }
        }
    }

    return $null
}

$resolvedAdminToken = Get-AdminTokenValue -Token $AdminToken
if ([string]::IsNullOrWhiteSpace($resolvedAdminToken)) {
    throw "ADMIN_TOKEN non está definido. Definea a variable de entorno ou escribe o valor no .env / .env.local."
}

$env:ADMIN_TOKEN = $resolvedAdminToken
$adminHeaders = @{ "X-Admin-Token" = $resolvedAdminToken }

Write-Host "[1/4] Crear clave API admin para alimentar shadow traffic..."
$createKeyBody = @{ name = $KeyName; role = "admin" } | ConvertTo-Json -Compress
$keyResp = Invoke-RestMethod -Uri "$ApiBase/admin/api-keys" -Method Post -Headers $adminHeaders -ContentType "application/json" -Body $createKeyBody
$apiKey = $keyResp.api_key
Write-Host "Clave creada: $($keyResp.prefix)"

$chatHeaders = @{ Authorization = "Bearer $apiKey"; "Content-Type" = "application/json" }
$successful = 0

Write-Host "[2/4] Acumulando $Samples peticións shadow con pausa de $DelayMs ms..."
for ($i = 1; $i -le $Samples; $i++) {
    $prompt = "shadow sample ${i}: Compare the safest way to diagnose a flaky Python API timeout with 2 root causes and a concise remediation checklist."
    $body = @{
        model = "CeltIA V4"
        messages = @(
            @{ role = "user"; content = $prompt }
        )
        max_tokens = $MaxTokens
    } | ConvertTo-Json -Depth 8

    try {
        Invoke-RestMethod -Uri "$ApiBase/v1/chat/completions" -Method Post -Headers $chatHeaders -Body $body -TimeoutSec 30 | Out-Null
        $successful++
    }
    catch {
        Write-Warning "request $i failed: $($_.Exception.Message)"
    }

    if ($i % 25 -eq 0) {
        Write-Host "  progress: $successful / $i requests OK"
        Start-Sleep -Seconds 2
    }
    else {
        Start-Sleep -Milliseconds $DelayMs
    }
}

Write-Host "[3/4] Comprobando readiness do rollout shadow..."
$ready = Invoke-RestMethod -Uri "$ApiBase/admin/decision-rollout-readiness?days=7" -Method Get -Headers $adminHeaders -TimeoutSec 30
$ready | ConvertTo-Json -Depth 8

Write-Host "[4/4] Resumo final: $successful peticións exitosas en shadow."
if ($ready.readiness.eligible_for_5pct_canary -eq $true) {
    Write-Host "Estado: ELIXIBLE para canary 5%."
    Write-Host "Seguinte paso: .\scripts\promote-cde-canary.ps1"
}
else {
    Write-Host "Estado: AINDA NON ELIXIBLE para canary 5%."
    Write-Host "Necesitas seguir acumulando shadow e comprobando readiness."
}
