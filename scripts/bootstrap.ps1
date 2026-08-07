# Bootstrap for native Windows (PowerShell 5.1+). Mirrors bootstrap.sh.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found; installing via astral.sh installer"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv python install 3.12

# nvidia-smi missing from PATH is not proof a GPU is absent: some driver
# packages (notably OEM laptop bundles) install the card without adding
# nvidia-smi to PATH. Fall back to a WMI adapter lookup so that case is
# reported honestly instead of silently routed into the CPU-only path.
$hasGpu = $false
$gpuNote = ""

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    try {
        nvidia-smi | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $hasGpu = $true
        } else {
            $gpuNote = "nvidia-smi exited with code $LASTEXITCODE"
        }
    } catch {
        $gpuNote = "nvidia-smi failed to run: $($_.Exception.Message)"
    }
} else {
    $adapter = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "NVIDIA" } |
        Select-Object -First 1
    if ($adapter) {
        $gpuNote = "NVIDIA adapter '$($adapter.Name)' is visible via WMI, but nvidia-smi is not on PATH"
    } else {
        $gpuNote = "nvidia-smi not found and no NVIDIA adapter is visible via WMI"
    }
}

if ($hasGpu) {
    Write-Host "NVIDIA GPU detected; syncing with the gpu extra (torch cu128)"
    uv sync --extra gpu
} else {
    Write-Host "No usable NVIDIA GPU detected ($gpuNote); syncing CPU-only"
    if ($gpuNote -match "WMI") {
        Write-Host "  A card is present but undetected: fix nvidia-smi's PATH, then re-run" -ForegroundColor Yellow
        Write-Host "  bootstrap, or run 'uv sync --extra gpu' manually once resolved." -ForegroundColor Yellow
    }
    uv sync
}

uv run pre-commit install

Write-Host ""
Write-Host "bootstrap complete. Next steps:"
Write-Host "  1. Copy-Item .env.example .env   # then paste your ANTHROPIC_API_KEY"
Write-Host "  2. make searxng                  # requires Docker Desktop"
Write-Host "  3. uv run quarry verify          # preflight, catches setup gaps before a real run"
if ($hasGpu) {
    Write-Host "  4. uv run quarry research `"your topic`""
} else {
    Write-Host "  4. make test                     # CPU-only suite; quarry research needs the GPU path above"
}
