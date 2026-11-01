"""RETRIEVE + RERANK per sub-question: ANN top-200, cross-encoder top-40.

Implemented in M5 (depends on store, embedder, reranker).
"""

from __future__ import annotations

import asyncio

from quarry_ldr.config import RetrieveSettings, SourceSettings
from quarry_ldr.gpu.embedder import Embedder
from quarry_ldr.gpu.reranker import Reranker, ScoredChunk
from quarry_ldr.index.store import VectorStore
from quarry_ldr.ingest.sources import adjusted_score, site_of
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


def diversify(scored: list[ScoredChunk], top_k: int, source: SourceSettings) -> list[ScoredChunk]:
    """The top_k by tier-adjusted score with at most rerank_site_limit per site.

    Long pages from one site can otherwise fill a sub-question's whole budget
    (measured: one site held 64 percent of a run's evidence). When limited
    sites cannot fill top_k, the next-best chunks fill it, so a thin corpus
    keeps its evidence. Raw scores stay on the returned items.
    """
    ranked = sorted(
        scored, key=lambda s: (-adjusted_score(s.chunk.url, s.score, source), s.chunk.chunk_id)
    )
    kept: list[ScoredChunk] = []
    overflow: list[ScoredChunk] = []
    per_site: dict[str, int] = {}
    for item in ranked:
        site = site_of(item.chunk.url)
        if per_site.get(site, 0) < source.rerank_site_limit:
            per_site[site] = per_site.get(site, 0) + 1
            kept.append(item)
        else:
            overflow.append(item)
    return (kept + overflow)[:top_k]


async def rerank_candidates(
    sub_question: SubQuestion,
    candidates: list[ScoredChunk],
    reranker: Reranker,
    settings: RetrieveSettings,
    source: SourceSettings,
) -> list[ScoredChunk]:
    """Cross-encoder scores for every candidate, then a diverse rerank_top_k."""
    if not candidates:
        return []
    scored = await reranker.rerank(
        sub_question.question, [c.chunk for c in candidates], len(candidates)
    )
    return diversify(scored, settings.rerank_top_k, source)
