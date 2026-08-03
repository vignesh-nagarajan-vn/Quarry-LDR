"""RETRIEVE + RERANK per sub-question: ANN top-200, cross-encoder top-40.

Implemented in M5 (depends on store, embedder, reranker).
"""

from __future__ import annotations

from quarry_ldr.config import RetrieveSettings
from quarry_ldr.gpu.embedder import Embedder
from quarry_ldr.gpu.reranker import Reranker, ScoredChunk
from quarry_ldr.index.store import VectorStore
from quarry_ldr.pipeline.plan import SubQuestion


async def retrieve_candidates(
    sub_question: SubQuestion,
    store: VectorStore,
    embedder: Embedder,
    settings: RetrieveSettings,
) -> list[ScoredChunk]:
    """ANN retrieval of ann_top_k candidates for one sub-question (score = -distance)."""
    raise NotImplementedError


async def rerank_candidates(
    sub_question: SubQuestion,
    candidates: list[ScoredChunk],
    reranker: Reranker,
    settings: RetrieveSettings,
) -> list[ScoredChunk]:
    """Cross-encoder rerank down to rerank_top_k."""
    raise NotImplementedError
