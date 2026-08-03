"""Generate the synthetic test corpus under tests/fixtures/.

Produces 40 HTML documents on one coherent invented topic, including
deliberate near-duplicates, one paywall stub, one JavaScript-only shell, one
malformed-encoding page, and one robots-disallowed URL, plus a hand-labeled
30-chunk relevance set for reranker regression. All content is synthetic;
nothing is scraped, because this repo goes public.

Deterministic: same seed, same bytes. Implemented in M3.
"""

from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError("implemented in milestone M3")


if __name__ == "__main__":
    sys.exit(main())
