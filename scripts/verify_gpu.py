"""Verify the GPU stack is actually usable, not merely detected.

Checks, in order:
  1. torch imports and CUDA is available;
  2. compute capability is at least (8, 0) so modern wheels have kernels
     (this repo was developed on an RTX 4060, sm_89; Blackwell sm_120 works
     with the pinned CUDA 12.8 wheel index);
  3. a real matmul executes on device and comes back finite, because
     torch.cuda.is_available() alone does not prove kernels exist;
  4. free VRAM is reported and compared against the configured arbiter budget.

Exit code 0 means the GPU path is trustworthy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MIN_CAPABILITY = (8, 0)
MATMUL_SIZE = 2048


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed.")
        print("fix:  uv sync --extra gpu")
        return 1

    if not torch.cuda.is_available():
        print("FAIL: torch imports but CUDA is not available.")
        print("fix:  check `nvidia-smi` works, driver supports CUDA 12.8+, and the")
        print("      installed torch wheel is a cu-series build (uv sync --extra gpu")
        print("      uses the pinned cu128 index).")
        return 1

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    free_b, total_b = torch.cuda.mem_get_info(0)
    free_mb, total_mb = free_b // 2**20, total_b // 2**20
    print(f"device:      {name}")
    print(f"capability:  {capability} (sm_{capability[0]}{capability[1]})")
    print(f"vram:        {free_mb} MB free / {total_mb} MB total")
    print(f"torch:       {torch.__version__} (cuda {torch.version.cuda})")

    if capability < MIN_CAPABILITY:
        print(f"FAIL: compute capability {capability} is below {MIN_CAPABILITY}.")
        print("fix:  this project targets Ampere (sm_80) or newer; older cards lack")
        print("      kernels in current wheels.")
        return 1

    try:
        a = torch.randn(MATMUL_SIZE, MATMUL_SIZE, device="cuda")
        b = torch.randn(MATMUL_SIZE, MATMUL_SIZE, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(c).all().item())
    except RuntimeError as exc:
        print(f"FAIL: on-device matmul raised: {exc}")
        print("fix:  the wheel has no kernels for this GPU architecture. For Blackwell")
        print("      (sm_120) you need CUDA 12.8+ wheels: the pinned cu128 index in")
        print("      pyproject.toml provides them; re-run `uv sync --extra gpu`.")
        return 1
    if not finite:
        print("FAIL: matmul produced non-finite values; the GPU path is not trustworthy.")
        return 1
    print(f"matmul:      {MATMUL_SIZE}x{MATMUL_SIZE} on-device ok, result finite")

    from quarry_ldr.config import load_config

    budget_mb = load_config().gpu.vram_budget_mb
    if budget_mb > total_mb:
        print(f"WARN: gpu.vram_budget_mb ({budget_mb}) exceeds total VRAM ({total_mb}).")
        print("fix:  lower gpu.vram_budget_mb to ~80 percent of total VRAM.")
        return 1
    if budget_mb > free_mb:
        print(f"WARN: only {free_mb} MB currently free, budget is {budget_mb} MB;")
        print("      close other GPU consumers before a run.")
    print(f"budget:      {budget_mb} MB arbiter budget fits this device")
    print("PASS: GPU path verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
