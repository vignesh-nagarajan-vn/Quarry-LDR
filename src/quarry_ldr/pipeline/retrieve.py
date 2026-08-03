"""RETRIEVE + RERANK per sub-question: ANN top-200, cross-encoder top-40.

Implemented in M5 (depends on store, embedder, reranker).
"""

from __future__ import annotations

import asyncio

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
    vec = await embedder.embed_query(sub_question.question)
    rows = await asyncio.to_thread(store.search, vec, settings.ann_top_k)
    return [ScoredChunk(chunk=row.chunk, score=-row.distance) for row in rows]


async def rerank_candidates(
    sub_question: SubQuestion,
    candidates: list[ScoredChunk],
    reranker: Reranker,
    settings: RetrieveSettings,
) -> list[ScoredChunk]:
    """Cross-encoder rerank down to rerank_top_k."""
    if not candidates:
        return []
    return await reranker.rerank(
        sub_question.question, [c.chunk for c in candidates], settings.rerank_top_k
    )
