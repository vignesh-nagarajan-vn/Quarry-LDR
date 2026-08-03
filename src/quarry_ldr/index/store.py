"""LanceDB vector store: idempotent adds, ANN search, schema-version checks.

The store's methods are synchronous (LanceDB's Python API is sync); pipeline
code calls them via ``asyncio.to_thread``.

Implemented in M5.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from quarry_ldr.ingest.chunk import Chunk


class RetrievedChunk(BaseModel):
    chunk: Chunk
    distance: float


class VectorStore:
    """One LanceDB table of chunks + vectors for a run's evidence."""

    def __init__(self, db_path: Path, dim: int) -> None:
        self.db_path = db_path
        self.dim = dim

    def open(self) -> None:
        """Connect and create-or-validate the table (SchemaMismatchError on drift)."""
        raise NotImplementedError

    def add(self, chunks: Sequence[Chunk], embeddings: NDArray[np.float32]) -> int:
        """Insert chunks with vectors; chunk_ids already present are skipped.
        Returns the number actually inserted."""
        raise NotImplementedError

    def search(self, query_vec: NDArray[np.float32], top_k: int) -> list[RetrievedChunk]:
        """ANN search, nearest first."""
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def has(self, chunk_id: str) -> bool:
        raise NotImplementedError
