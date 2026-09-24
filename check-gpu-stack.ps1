# check-gpu-stack.ps1
# Diagnóstico Windows + WSL2 + NVIDIA + Docker + RTX 3060 Ti
# SOLO COMPRUEBA. NO INSTALA NI MODIFICA NADA.
# ============================================================

$ErrorActionPreference = "Continue"

function Test-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Name
    Write-Host "============================================================"

    try {
        & $Action

        $exitCode = $LASTEXITCODE

        if ($null -eq $exitCode -or $exitCode -eq 0) {
            Write-Host "[OK] $Name" -ForegroundColor Green
            return $true
        }
        else {
            Write-Host "[ERROR] $Name (exit code $exitCode)" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "[ERROR] $Name" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Yellow
        return $false
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "  DIAGNOSTICO RTX 3060 Ti - Windows + WSL2 + Docker"
Write-Host "============================================================"
Write-Host ""
Write-Host "Este script NO instala ni modifica componentes."
Write-Host ""

# ------------------------------------------------------------
# 1. Windows
# ------------------------------------------------------------

Test-Step "1. Windows / PowerShell" {
    Write-Host "Windows:"
    Get-CimInstance Win32_OperatingSystem |
        Select-Object Caption, Version, BuildNumber |
        Format-List

    Write-Host "PowerShell: $($PSVersionTable.PSVersion)"
}

# ------------------------------------------------------------
# 2. NVIDIA en Windows
# ------------------------------------------------------------

Test-Step "2. NVIDIA GPU en Windows" {
    $gpu = Get-CimInstance Win32_VideoController |
        Where-Object { $_.Name -match "NVIDIA" }

    if (-not $gpu) {
        throw "No se ha encontrado una GPU NVIDIA en Windows."
    }

    $gpu | Select-Object Name, DriverVersion, VideoModeDescription |
        Format-List
}

# ------------------------------------------------------------
# 3. nvidia-smi en Windows
# ------------------------------------------------------------

Test-Step "3. nvidia-smi en Windows" {
    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue

    if (-not $nvidiaSmi) {
        throw "nvidia-smi.exe no está disponible en PATH."
    }

    Write-Host "Ejecutable: $($nvidiaSmi.Source)"
    nvidia-smi
}

# ------------------------------------------------------------
# 4. WSL instalado
# ------------------------------------------------------------

Test-Step "4. WSL instalado y disponible" {
    wsl.exe --status
}

# ------------------------------------------------------------
# 5. Distribuciones WSL
# ------------------------------------------------------------

Test-Step "5. Distribuciones WSL" {
    wsl.exe --list --verbose
}

# ------------------------------------------------------------
# 6. Comprobar versión por defecto
# ------------------------------------------------------------

Test-Step "6. WSL2 como versión por defecto" {
    $output = wsl.exe --list --verbose 2>&1
    $text = $output -join "`n"

    Write-Host $text

    if ($text -notmatch "VERSION") {
        Write-Host "No se ha podido determinar la versión de las distribuciones."
    }

    if ($text -match "2\s+\*?\s*$|VERSION\s+STATE\s+VERSION") {
        Write-Host "Se ha detectado una referencia a WSL2 en la salida."
    }
}

# ------------------------------------------------------------
# 7. Selección de distribución WSL
# ------------------------------------------------------------

Write-Host ""
Write-Host "============================================================"
Write-Host "7. Selección de distribución WSL"
Write-Host "============================================================"

$wslRaw = wsl.exe --list --quiet 2>$null
$distros = @(
    $wslRaw |
        ForEach-Object { ($_ -replace "`0", "").Trim() } |
        Where-Object { $_ }
)

if (-not $distros) {
    Write-Host "No hay distribuciones WSL instaladas." -ForegroundColor Yellow
}
else {
    Write-Host "Distribuciones detectadas:" -ForegroundColor Cyan
    $distros | ForEach-Object { Write-Host " - $_" }

    $selectedDistro = $distros[0]
    Write-Host "Distribución elegida para comprobaciones: $selectedDistro"

    Test-Step "7a. Información básica de la distro WSL" {
        wsl.exe -d $selectedDistro -- uname -a
    }

    Test-Step "7b. Sistema operativo dentro de WSL" {
        wsl.exe -d $selectedDistro -- cat /etc/os-release
    }
}

# ------------------------------------------------------------
# 8. NVIDIA dentro de WSL
# ------------------------------------------------------------

Test-Step "8. NVIDIA en WSL" {
    if (-not $distros) {
        throw "No hay distribuciones WSL para comprobar NVIDIA."
    }

    $selectedDistro = $distros[0]

    Write-Host "Comprobando NVIDIA desde WSL: $selectedDistro"

    $nvidiaCheck = wsl.exe -d $selectedDistro -- sh -lc "which nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo 'nvidia-smi no disponible dentro de WSL'"
    Write-Host ($nvidiaCheck -join "`n")

    if (($nvidiaCheck -join "`n") -match "no disponible dentro de WSL") {
        Write-Host "" 
        Write-Host "[DIAGNOSTICO] WSL sin soporte de GPU" -ForegroundColor Yellow
        Write-Host "Causa probable: la distribución WSL no tiene acceso a la GPU NVIDIA." -ForegroundColor Yellow
        Write-Host "Revisa: driver NVIDIA en Windows, WSL2 habilitado, CUDA en la distro y soporte de GPU en Docker Desktop." -ForegroundColor Yellow
        throw "nvidia-smi no está disponible dentro de la distribución WSL seleccionada."
    }
}

# ------------------------------------------------------------
# 9. Docker Desktop / CLI
# ------------------------------------------------------------

Test-Step "9. Docker CLI disponible" {
    $dockerCmd = Get-Command docker.exe -ErrorAction SilentlyContinue

    if (-not $dockerCmd) {
        throw "docker.exe no está disponible en PATH."
    }

    Write-Host "Ejecutable: $($dockerCmd.Source)"
    docker version --format "{{.Server.Version}}"
}

Test-Step "10. Docker Desktop / daemon funcionando" {
    docker info
}

Test-Step "10b. Docker Desktop con integración NVIDIA" {
    $dockerInfo = docker info 2>&1
    $dockerText = $dockerInfo -join "`n"
    Write-Host $dockerText

    if ($dockerText -match "nvidia|NVIDIA") {
        Write-Host "Se ha detectado referencia a NVIDIA dentro del daemon de Docker." -ForegroundColor Green
    }
    else {
        Write-Host "" 
        Write-Host "[DIAGNOSTICO] Docker Desktop sin integración NVIDIA" -ForegroundColor Yellow
        Write-Host "Suele significar que Docker Desktop no está configurado para acceder a la GPU mediante WSL2/Windows." -ForegroundColor Yellow
        Write-Host "Revisa: Docker Desktop > Settings > Resources > WSL Integration y soporte de GPU." -ForegroundColor Yellow
    }
}

# ------------------------------------------------------------
# 11. NVIDIA Container Toolkit y compatibilidad
# ------------------------------------------------------------

Test-Step "11. NVIDIA Container Toolkit disponible dentro de WSL" {
    if (-not $distros) {
        throw "No hay distribuciones WSL para comprobar el toolkit NVIDIA."
    }

    $selectedDistro = $distros[0]

    $toolkitCheck = wsl.exe -d $selectedDistro -- sh -lc "(nvidia-container-cli --version || nvidia-ctk --version || echo 'nvidia-container-toolkit no instalado') 2>&1"
    $toolkitText = $toolkitCheck -join "`n"
    Write-Host $toolkitText

    if ($toolkitText -match "no instalado|not found|command not found") {
        Write-Host "" 
        Write-Host "[DIAGNOSTICO] NVIDIA Container Toolkit no presente" -ForegroundColor Yellow
        Write-Host "Sin el toolkit, Docker no puede montar contenedores con acceso GPU NVIDIA." -ForegroundColor Yellow
        Write-Host "En Linux/WSL esto suele requerir instalar nvidia-container-toolkit + el runtime nvidia en Docker." -ForegroundColor Yellow
    }
}

# ------------------------------------------------------------
# 12. Docker con GPU (solo comprobación)
# ------------------------------------------------------------

Test-Step "12. Docker puede ver GPU NVIDIA" {
    $dockerGpuCheck = docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi 2>&1
    $dockerGpuText = $dockerGpuCheck -join "`n"
    Write-Host $dockerGpuText

    if ($dockerGpuText -match "No devices were found|could not select device driver|unknown flag: --gpus|docker: Error response|nvidia-smi: command not found|could not select device driver") {
        Write-Host "" 
        Write-Host "[DIAGNOSTICO] Docker Desktop sin integración NVIDIA o NVIDIA Container Toolkit no compatible" -ForegroundColor Yellow
        Write-Host "Esto suele indicar que faltan los componentes necesarios para acceder a la GPU desde contenedores." -ForegroundColor Yellow
        Write-Host "Verifica: driver NVIDIA, WSL2 con GPU, Docker Desktop con WSL integration, y toolkit NVIDIA compatible." -ForegroundColor Yellow
        throw "Docker no puede usar la GPU NVIDIA en este sistema."
    }
}

# ------------------------------------------------------------
# 13. Resumen final
# ------------------------------------------------------------

Write-Host ""
Write-Host "============================================================"
Write-Host "RESUMEN"
Write-Host "============================================================"
Write-Host ""
Write-Host "Este diagnóstico solo comprueba la disponibilidad del stack NVIDIA + WSL2 + Docker."
Write-Host "No intenta instalar ni modificar drivers, WSL ni Docker."
Write-Host ""
Write-Host "Si falla la comprobación de 'nvidia-smi' dentro de WSL o Docker con --gpus all,"
Write-Host "el problema real suele estar en una de estas capas:"
Write-Host "  - GPU NVIDIA no detectada por Windows"
Write-Host "  - Driver NVIDIA no instalado o desactualizado"
Write-Host "  - WSL sin soporte de GPU"
Write-Host "  - Docker Desktop sin integración NVIDIA"
Write-Host "  - NVIDIA Container Toolkit no presente o no compatible"
Write-Host ""
Write-Host "Diagnóstico concreto:"
Write-Host "  * Si falla 'nvidia-smi' dentro de WSL => WSL sin soporte de GPU"
Write-Host "  * Si docker info no referencia a NVIDIA => Docker Desktop sin integración NVIDIA"
Write-Host "  * Si falta nvidia-container-cli / nvidia-ctk => NVIDIA Container Toolkit no presente"
Write-Host "  * Si docker run --gpus all falla => integración NVIDIA o toolkit incompatible"
Write-Host ""
