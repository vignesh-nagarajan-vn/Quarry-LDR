"""Near-duplicate removal: SimHash shingling plus embedding cosine similarity.

News and blogs repeat each other; expect to drop 30 to 50 percent of chunks.
A chunk is a duplicate of an earlier kept chunk when EITHER its 64-bit SimHash
is within ``simhash_hamming_max`` bits OR its embedding cosine similarity is
at or above ``cosine_threshold``.

Implemented in M5.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from quarry_ldr.config import DedupSettings
from quarry_ldr.ingest.chunk import Chunk


def simhash64(text: str, shingle_size: int = 5) -> int:
    """64-bit SimHash over word shingles of ``shingle_size`` words."""
    raise NotImplementedError


def hamming(a: int, b: int) -> int:
    """Hamming distance between two 64-bit hashes."""
    raise NotImplementedError


class DedupResult(BaseModel):
    """Indices refer to positions in the input sequence."""

    kept: list[int]
    dropped: dict[int, int] = Field(default_factory=dict)  # dropped_idx -> kept_idx it duplicates
    n_input: int
    n_kept: int

    @property
    def drop_rate(self) -> float:
        return 0.0 if self.n_input == 0 else 1.0 - (self.n_kept / self.n_input)


def dedup_chunks(
    chunks: Sequence[Chunk],
    embeddings: NDArray[np.float32] | None,
    settings: DedupSettings,
) -> DedupResult:
    """Greedy first-wins dedup in input order.

    ``embeddings`` rows must align with ``chunks`` and be L2-normalized; pass
    None to run SimHash-only (used before embeddings exist).
    """
    raise NotImplementedError
