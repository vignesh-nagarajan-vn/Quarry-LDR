"""Go/no-go audit before this repo is made public.

Scans the full git history for secret patterns, checks every tracked file for
absolute personal paths and identifiers, verifies license presence, verifies
fixtures are synthetic (provenance marker emitted by make_fixtures.py),
and enforces the no-em-dash rule in README.md, CLAUDE.md, and COMMIT.md.
Prints a verdict and exits nonzero on no-go. Implemented in M10.
"""

from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError("implemented in milestone M10")


if __name__ == "__main__":
    sys.exit(main())
