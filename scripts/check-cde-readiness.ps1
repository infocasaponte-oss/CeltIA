param(
    [string]$ApiBase = "http://127.0.0.1:8081",
    [int]$Days = 7
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($env:ADMIN_TOKEN)) {
    throw "ADMIN_TOKEN non está definido no ambiente."
}

$headers = @{ "X-Admin-Token" = $env:ADMIN_TOKEN }
$uri = "$ApiBase/admin/decision-rollout-readiness?days=$Days"
$result = Invoke-RestMethod -Uri $uri -Headers $headers -Method Get -TimeoutSec 15

$result | ConvertTo-Json -Depth 8

if ($result.readiness.eligible_for_5pct_canary -eq $true) {
    Write-Host "CDE shadow: ELIXIBLE para canary 5%."
    exit 0
}

Write-Host "CDE shadow: NON elixible para canary 5%."
if ($result.readiness.reasons) {
    Write-Host ("Razóns: " + ($result.readiness.reasons -join ", "))
}
exit 2
