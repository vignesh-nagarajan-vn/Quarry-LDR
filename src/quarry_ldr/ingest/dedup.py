"""Near-duplicate removal: SimHash shingling plus embedding cosine similarity.

News and blogs repeat each other; expect to drop 30 to 50 percent of chunks.
A chunk is a duplicate of an earlier kept chunk when EITHER its 64-bit SimHash
is within ``simhash_hamming_max`` bits OR its embedding cosine similarity is
at or above ``cosine_threshold``.

Implemented in M5.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from quarry_ldr.config import DedupSettings
from quarry_ldr.ingest.chunk import Chunk


def _shingle_hash(shingle: str) -> int:
    """Stable 64-bit hash of a shingle string via blake2b digest_size=8."""
    digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def simhash64(text: str, shingle_size: int = 5) -> int:
    """64-bit SimHash over word shingles of ``shingle_size`` words."""
    words = text.lower().split()
    if not words:
        return 0

    if len(words) < shingle_size:
        shingles = [" ".join(words)]
    else:
        shingles = [
            " ".join(words[i : i + shingle_size]) for i in range(len(words) - shingle_size + 1)
        ]

    bit_weights = [0] * 64
    for shingle in shingles:
        h = _shingle_hash(shingle)
        for bit in range(64):
            if (h >> bit) & 1:
                bit_weights[bit] += 1
            else:
                bit_weights[bit] -= 1

    result = 0
    for bit in range(64):
        if bit_weights[bit] > 0:
            result |= 1 << bit
    return result


def hamming(a: int, b: int) -> int:
    """Hamming distance between two 64-bit hashes."""
    return (a ^ b).bit_count()


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
    n = len(chunks)
    kept: list[int] = []
    kept_simhashes: list[int] = []
    dropped: dict[int, int] = {}

    kept_embeddings: NDArray[np.float32] | None = None
    if embeddings is not None and n > 0:
        kept_embeddings = np.empty((n, embeddings.shape[1]), dtype=embeddings.dtype)

    for i, chunk in enumerate(chunks):
        sim_i = simhash64(chunk.text)
        n_kept_so_far = len(kept)

        cos_sims: NDArray[np.float32] | None = None
        if embeddings is not None and kept_embeddings is not None and n_kept_so_far > 0:
            cos_sims = kept_embeddings[:n_kept_so_far] @ embeddings[i]

        dup_of: int | None = None
        for pos, j in enumerate(kept):
            simhash_match = (
                sim_i != 0
                and kept_simhashes[pos] != 0
                and hamming(sim_i, kept_simhashes[pos]) <= settings.simhash_hamming_max
            )
            cosine_match = (
                cos_sims is not None and float(cos_sims[pos]) >= settings.cosine_threshold
            )
            if simhash_match or cosine_match:
                dup_of = j
                break

        if dup_of is not None:
            dropped[i] = dup_of
        else:
            if embeddings is not None and kept_embeddings is not None:
                kept_embeddings[len(kept)] = embeddings[i]
            kept.append(i)
            kept_simhashes.append(sim_i)

    return DedupResult(kept=kept, dropped=dropped, n_input=n, n_kept=len(kept))
