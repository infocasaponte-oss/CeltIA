param(
    [Parameter(Mandatory=$true)]
    [ValidateSet(5,20,50,100)]
    [int]$CurrentPercent,
    [string]$ApiBase = "http://127.0.0.1:8081",
    [int]$Days = 7
)

$ErrorActionPreference = "Stop"
$checker = Join-Path $PSScriptRoot "check-cde-canary.ps1"
& $checker -Percent $CurrentPercent -ApiBase $ApiBase -Days $Days
if ($LASTEXITCODE -ne 0) {
    throw "A etapa canary actual non pasa readiness. Non se modificou o rollout."
}

$next = switch ($CurrentPercent) {
    5 { 20 }
    20 { 50 }
    50 { 100 }
    100 { $null }
}

if ($null -eq $next) {
    [Environment]::SetEnvironmentVariable("DECISION_ROUTING_MODE", "cde", "User")
    [Environment]::SetEnvironmentVariable("DECISION_CDE_ROLLOUT_PERCENT", "100", "User")
    $env:DECISION_ROUTING_MODE = "cde"
    $env:DECISION_CDE_ROLLOUT_PERCENT = "100"
    Write-Host "Promoción final preparada: DECISION_ROUTING_MODE=cde."
} else {
    [Environment]::SetEnvironmentVariable("DECISION_ROUTING_MODE", "canary", "User")
    [Environment]::SetEnvironmentVariable("DECISION_CDE_ROLLOUT_PERCENT", "$next", "User")
    $env:DECISION_ROUTING_MODE = "canary"
    $env:DECISION_CDE_ROLLOUT_PERCENT = "$next"
    Write-Host "Seguinte etapa preparada: canary $next%."
}

Write-Host "Reinicia a API para aplicar o cambio."
Write-Host "Rollback inmediato: scripts\rollback-cde.ps1"
