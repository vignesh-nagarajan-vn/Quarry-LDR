"""Verify the GPU stack is actually usable, not merely detected.

Checks, in order:
  1. torch imports and CUDA is available;
  2. compute capability is at least (8, 0) so modern wheels have kernels
     (this repo was developed on an RTX 4060, sm_89; Blackwell sm_120 works
     with CUDA 12.8+ wheels);
  3. a real matmul executes on device and comes back finite, because
     torch.cuda.is_available() alone does not prove kernels exist;
  4. free VRAM is reported and compared against the configured budget.

Exit code 0 means the GPU path is trustworthy. Implemented in M2.
"""

from __future__ import annotations

import sys

MIN_CAPABILITY = (8, 0)


def main() -> int:
    raise NotImplementedError("implemented in milestone M2")


if __name__ == "__main__":
    sys.exit(main())
