param(
    [Parameter(Mandatory=$true)]
    [ValidateSet(5,20,50,100)]
    [int]$Percent,
    [string]$ApiBase = "http://127.0.0.1:8081",
    [int]$Days = 7
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($env:ADMIN_TOKEN)) {
    throw "ADMIN_TOKEN non está definido no ambiente."
}

$headers = @{ "X-Admin-Token" = $env:ADMIN_TOKEN }
$uri = "$ApiBase/admin/decision-canary-readiness?percent=$Percent&days=$Days"
$result = Invoke-RestMethod -Uri $uri -Headers $headers -Method Get -TimeoutSec 15
$result | ConvertTo-Json -Depth 8

if ($result.readiness.eligible_for_next_stage -eq $true) {
    Write-Host "Canary $Percent%: ELIXIBLE para a seguinte etapa."
    exit 0
}

Write-Host "Canary $Percent%: NON elixible para a seguinte etapa."
if ($result.readiness.reasons) {
    Write-Host ("Razóns: " + ($result.readiness.reasons -join ", "))
}
exit 2
