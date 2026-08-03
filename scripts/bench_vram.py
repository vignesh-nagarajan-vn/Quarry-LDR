"""Measure real VRAM footprints and throughput for every registered model.

Loads each model alone under the arbiter, records torch.cuda.mem_get_info
before and after, runs a small workload for tokens/sec (the laptop thermal
story), and prints a table to paste into DECISIONS.md and config
gpu.footprints_mb. Implemented in M10.
"""

from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError("implemented in milestone M10")


if __name__ == "__main__":
    sys.exit(main())
