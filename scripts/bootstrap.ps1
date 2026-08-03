# Bootstrap for native Windows (PowerShell 5.1+). Mirrors bootstrap.sh.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found; installing via astral.sh installer"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv python install 3.12

$hasGpu = $false
try { nvidia-smi | Out-Null; $hasGpu = $true } catch {}

if ($hasGpu) {
    Write-Host "NVIDIA GPU detected; syncing with the gpu extra (torch cu128)"
    uv sync --extra gpu
} else {
    Write-Host "No NVIDIA GPU detected; syncing CPU-only (tests do not need a GPU)"
    uv sync
}

uv run pre-commit install

Write-Host ""
Write-Host "bootstrap complete. Next steps:"
Write-Host "  1. Copy-Item .env.example .env   # then paste your ANTHROPIC_API_KEY"
Write-Host "  2. make searxng                  # requires Docker Desktop"
Write-Host "  3. uv run quarry verify          # preflight"
Write-Host "  4. uv run quarry research `"your topic`""
