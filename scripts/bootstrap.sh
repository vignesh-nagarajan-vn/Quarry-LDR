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

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected; syncing with the gpu extra (torch cu128)"
    uv sync --extra gpu
else
    echo "No NVIDIA GPU detected; syncing CPU-only (tests do not need a GPU)"
    uv sync
fi

uv run pre-commit install

echo ""
echo "bootstrap complete. Next steps:"
echo "  1. cp .env.example .env   # then paste your ANTHROPIC_API_KEY"
echo "  2. make searxng           # requires Docker Desktop / Docker Engine"
echo "  3. uv run quarry verify   # preflight"
echo "  4. uv run quarry research \"your topic\""
