$ErrorActionPreference = "Stop"

[Environment]::SetEnvironmentVariable("DECISION_ROUTING_MODE", "legacy", "User")
[Environment]::SetEnvironmentVariable("DECISION_CDE_ROLLOUT_PERCENT", "0", "User")

$env:DECISION_ROUTING_MODE = "legacy"
$env:DECISION_CDE_ROLLOUT_PERCENT = "0"

Write-Host "Rollback CDE preparado: DECISION_ROUTING_MODE=legacy, DECISION_CDE_ROLLOUT_PERCENT=0."
Write-Host "Reinicia a API para aplicar."
