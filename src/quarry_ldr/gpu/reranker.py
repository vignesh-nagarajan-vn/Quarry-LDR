"""Cross-encoder reranking: 200 ANN candidates down to 40, on the GPU.

This is the single highest-leverage quality lever in the system. Default
model: BAAI/bge-reranker-v2-m3.

Implemented in M5.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.arbiter import VramArbiter
from quarry_ldr.ingest.chunk import Chunk


class ScoredChunk(BaseModel):
    """A chunk with a relevance score (higher is more relevant)."""

    chunk: Chunk
    score: float


class Reranker:
    """Scores (query, chunk) pairs under ``arbiter.acquire("reranker")``."""

    ARBITER_NAME = "reranker"

    def __init__(self, cfg: QuarryConfig, arbiter: VramArbiter) -> None:
        self.cfg = cfg
        self.arbiter = arbiter
        self.model_id = cfg.models.reranker
        self.batch_size = cfg.gpu.rerank_batch_size

    async def rerank(self, query: str, chunks: Sequence[Chunk], top_k: int) -> list[ScoredChunk]:
        """Score every chunk against the query, return the top_k best, sorted."""
        raise NotImplementedError
