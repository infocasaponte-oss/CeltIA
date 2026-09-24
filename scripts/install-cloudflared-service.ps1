#Requires -RunAsAdministrator
<#
  Repara el servicio de Windows "Cloudflared" para que conecte SIEMPRE al túnel celtiav2 al arrancar el PC.

  Ejecutar en una PowerShell como Administrador (desde cualquier carpeta):
      & "D:\CeltIA V4-2B\scripts\install-cloudflared-service.ps1"

  Pide el token del túnel (entrada oculta). Si pulsas solo Enter, intenta obtenerlo de la API de Cloudflare
  usando CLOUDFLARE_TUNNEL_TOKEN de .env.local cuando ese valor es una Global API Key (cfk_...).
  El token nunca se imprime. Antes de guardarlo se comprueba que pertenece al túnel celtiav2.
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$tunnelId = "acf924f6-bb77-4898-a2d3-ed332f46122c"   # celtiav2
$accountId = "5635fea886f0d55b100a808a9cac891b"
$tokenDir = "C:\ProgramData\cloudflared"

function Get-TunnelIdFromToken([string]$t) {
    try {
        $b = $t.Trim().Replace('-', '+').Replace('_', '/')
        while ($b.Length % 4) { $b += '=' }
        return ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($b)) | ConvertFrom-Json).t
    } catch { return $null }
}

$secure = Read-Host "Pega el token del túnel celtiav2 (o Enter para obtenerlo automáticamente)" -AsSecureString
$token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)).Trim()

if (-not $token) {
    $line = Get-Content (Join-Path $root ".env.local") | Where-Object { $_ -match '^\s*CLOUDFLARE_TUNNEL_TOKEN\s*=' } | Select-Object -First 1
    if (-not $line) { throw "No has pegado ningún token y falta CLOUDFLARE_TUNNEL_TOKEN en .env.local" }
    $value = ($line -split '=', 2)[1].Trim().Trim('"').Trim("'")
    if ($value.StartsWith("cfk_")) {
        $h = @{ "X-Auth-Email" = "infocasaponte@gmail.com"; "X-Auth-Key" = $value }
        $token = (Invoke-RestMethod "https://api.cloudflare.com/client/v4/accounts/$accountId/cfd_tunnel/$tunnelId/token" -Headers $h).result
    } else { $token = $value }
}
if (-not $token) { throw "No se pudo obtener el token del túnel" }

$id = Get-TunnelIdFromToken $token
if ($id -ne $tunnelId) { throw "Ese token no es de celtiav2 (apunta a: $id). No se ha cambiado nada." }

New-Item -ItemType Directory -Force $tokenDir | Out-Null
Set-Content -Path (Join-Path $tokenDir "token") -Value $token -NoNewline
Restart-Service Cloudflared
Start-Sleep 8
Get-Service Cloudflared | Format-Table Name, Status, StartType -AutoSize
try {
    $r = Invoke-WebRequest "https://celtiaia.com/health" -UseBasicParsing -TimeoutSec 20
    Write-Host "https://celtiaia.com/health -> $($r.StatusCode)"
} catch { Write-Host "Servicio reiniciado, pero celtiaia.com aún no responde: espera unos segundos y prueba de nuevo." }
