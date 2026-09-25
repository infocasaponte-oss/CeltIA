param(
    [string]$ApiBase = "http://127.0.0.1:8081",
    [int]$Days = 7
)

$ErrorActionPreference = "Stop"

$checker = Join-Path $PSScriptRoot "check-cde-readiness.ps1"
& $checker -ApiBase $ApiBase -Days $Days
if ($LASTEXITCODE -ne 0) {
    throw "Readiness non pasa. Non se modificou o rollout."
}

[Environment]::SetEnvironmentVariable("DECISION_ROUTING_MODE", "canary", "User")
[Environment]::SetEnvironmentVariable("DECISION_CDE_ROLLOUT_PERCENT", "5", "User")

$env:DECISION_ROUTING_MODE = "canary"
$env:DECISION_CDE_ROLLOUT_PERCENT = "5"

Write-Host "Canary 5% preparado no ambiente de usuario."
Write-Host "Reinicia a API para aplicar: DECISION_ROUTING_MODE=canary, DECISION_CDE_ROLLOUT_PERCENT=5."
Write-Host "Rollback: scripts\rollback-cde.ps1"
