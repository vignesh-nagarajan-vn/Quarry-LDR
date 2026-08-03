"""End-to-end smoke test on real infrastructure with a hard $0.50 cost cap.

Runs the actual pipeline (real GPU, real SearXNG, real API) on a trivial
topic, prints the ledger, checks the report has resolvable citations, and
exits nonzero if anything is off. This is the one script the human runs at
the end. Implemented in M10.
"""

from __future__ import annotations

import sys

COST_CAP_USD = 0.50


def main() -> int:
    raise NotImplementedError("implemented in milestone M10")


if __name__ == "__main__":
    sys.exit(main())
