"""Cross-encoder reranking: 200 ANN candidates down to 40, on the GPU.

This is the single highest-leverage quality lever in the system. Default
model: BAAI/bge-reranker-v2-m3.

Implemented in M5.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.arbiter import ModelSpec, VramArbiter
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
        arbiter.register(
            ModelSpec(
                name=self.ARBITER_NAME,
                footprint_mb=cfg.gpu.footprints_mb.get("reranker", 2128),
                loader=self._load_model,
                unloader=self._unload_model,
            )
        )

    def _load_model(self) -> Any:
        """Lazily import sentence-transformers so CPU-only environments never
        pay the import cost unless a rerank actually happens on a GPU box."""
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_id, device="cuda")

    def _unload_model(self, model: Any) -> None:
        del model

    async def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Score arbitrary (text, text) pairs, one batched pass.

        Serves two callers: rerank's (query, chunk) scoring and the VERIFY
        stage's (sentence, cited evidence) entailment signal.
        """
        if not pairs:
            return []
        async with self.arbiter.acquire(self.ARBITER_NAME) as model:
            scores = await asyncio.to_thread(model.predict, list(pairs), batch_size=self.batch_size)
        return [float(score) for score in scores]

    async def rerank(self, query: str, chunks: Sequence[Chunk], top_k: int) -> list[ScoredChunk]:
        """Score every chunk against the query, return the top_k best, sorted."""
        if not chunks:
            return []
        scores = await self.score_pairs([(query, chunk.text) for chunk in chunks])
        scored = [
            ScoredChunk(chunk=chunk, score=score)
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        # list.sort with reverse=True stays stable: equal-score chunks keep
        # their original input order rather than being reversed.
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:top_k]
