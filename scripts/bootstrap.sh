#!/usr/bin/env bash
# Bootstrap for Linux / WSL2 / Git Bash. Installs uv if missing, syncs the
# environment (GPU extra when an NVIDIA GPU is visible), installs pre-commit
# hooks, and prints next steps.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing via astral.sh installer"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.12

# nvidia-smi missing from PATH is not proof a GPU is absent. Fall back to the
# /proc driver signature so that case is reported honestly instead of
# silently routed into the CPU-only path.
has_gpu=0
gpu_note=""
if command -v nvidia-smi >/dev/null 2>&1; then
    has_gpu=1
elif [ -e /proc/driver/nvidia/version ]; then
    gpu_note="an NVIDIA driver is loaded (/proc/driver/nvidia/version), but nvidia-smi is not on PATH"
else
    gpu_note="nvidia-smi not found and no NVIDIA driver signature at /proc/driver/nvidia/version"
fi

if [ "$has_gpu" = "1" ]; then
    echo "NVIDIA GPU detected; syncing with the gpu extra (torch cu128)"
    uv sync --extra gpu
else
    echo "No usable NVIDIA GPU detected ($gpu_note); syncing CPU-only"
    uv sync
fi

uv run pre-commit install

echo ""
echo "bootstrap complete. Next steps:"
echo "  1. cp .env.example .env   # then paste your ANTHROPIC_API_KEY"
echo "  2. make searxng           # requires Docker Desktop / Docker Engine"
echo "  3. uv run quarry verify   # preflight, catches setup gaps before a real run"
if [ "$has_gpu" = "1" ]; then
    echo "  4. uv run quarry research \"your topic\""
else
    echo "  4. make test              # CPU-only suite; quarry research needs the GPU path above"
fi
