"""retrieve_candidates / rerank_candidates: CPU-only tests against duck-typed
fakes for the store, embedder, and reranker, cast to their annotated types."""

from __future__ import annotations

import typing
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from quarry_ldr.config import RetrieveSettings
from quarry_ldr.gpu.embedder import Embedder
from quarry_ldr.gpu.reranker import Reranker, ScoredChunk
from quarry_ldr.index.store import RetrievedChunk, VectorStore
from quarry_ldr.ingest.chunk import Chunk
from quarry_ldr.pipeline.plan import SubQuestion
from quarry_ldr.pipeline.retrieve import rerank_candidates, retrieve_candidates


def make_chunk(position: int, text: str = "text") -> Chunk:
    return Chunk(
        chunk_id=f"chunk-{position}",
        url="https://example.com/doc",
        text=text,
        token_count=max(1, len(text) // 4),
        position=position,
        start_char=0,
        end_char=len(text),
    )


def make_sub_question(question: str = "What is the round-trip efficiency?") -> SubQuestion:
    return SubQuestion(
        id="sq01",
        question=question,
        queries=["a", "b"],
        success_criterion="answers with a percentage",
    )


class FakeEmbedder:
    """Duck-typed stand-in for Embedder; records the query text it received."""

    def __init__(self, vec: NDArray[np.float32] | None = None) -> None:
        self.vec = vec if vec is not None else np.zeros(4, dtype=np.float32)
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> NDArray[np.float32]:
        self.queries.append(text)
        return self.vec


class FakeStore:
    """Duck-typed stand-in for VectorStore; records (vec, top_k) per search()."""

    def __init__(self, rows: list[RetrievedChunk]) -> None:
        self.rows = rows
        self.search_calls: list[tuple[Any, int]] = []

    def search(self, query_vec: NDArray[np.float32], top_k: int) -> list[RetrievedChunk]:
        self.search_calls.append((query_vec, top_k))
        return self.rows[:top_k]


class FakeReranker:
    """Duck-typed stand-in for Reranker; records every rerank() invocation."""

    def __init__(self, result: list[ScoredChunk] | None = None) -> None:
        self.result = result if result is not None else []
        self.calls: list[tuple[str, list[Chunk], int]] = []

    async def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[ScoredChunk]:
        self.calls.append((query, list(chunks), top_k))
        return self.result


async def test_retrieve_candidates_negates_distance_into_score() -> None:
    sq = make_sub_question()
    rows = [RetrievedChunk(chunk=make_chunk(0), distance=0.1)]
    store = FakeStore(rows)
    embedder = FakeEmbedder()
    settings = RetrieveSettings(ann_top_k=200, rerank_top_k=40)

    result = await retrieve_candidates(
        sq, typing.cast(VectorStore, store), typing.cast(Embedder, embedder), settings
    )

    assert result == [ScoredChunk(chunk=rows[0].chunk, score=-0.1)]


async def test_retrieve_candidates_passes_ann_top_k_to_search() -> None:
    sq = make_sub_question()
    rows = [RetrievedChunk(chunk=make_chunk(i), distance=float(i)) for i in range(5)]
    store = FakeStore(rows)
    embedder = FakeEmbedder()
    settings = RetrieveSettings(ann_top_k=3, rerank_top_k=40)

    await retrieve_candidates(
        sq, typing.cast(VectorStore, store), typing.cast(Embedder, embedder), settings
    )

    assert store.search_calls == [(embedder.vec, 3)]


async def test_retrieve_candidates_embeds_sub_question_text() -> None:
    sq = make_sub_question(question="How efficient is the sand battery?")
    store = FakeStore([])
    embedder = FakeEmbedder()
    settings = RetrieveSettings()

    await retrieve_candidates(
        sq, typing.cast(VectorStore, store), typing.cast(Embedder, embedder), settings
    )

    assert embedder.queries == ["How efficient is the sand battery?"]


async def test_retrieve_candidates_preserves_store_ordering() -> None:
    sq = make_sub_question()
    rows = [
        RetrievedChunk(chunk=make_chunk(0), distance=0.5),
        RetrievedChunk(chunk=make_chunk(1), distance=0.1),
        RetrievedChunk(chunk=make_chunk(2), distance=0.9),
    ]
    store = FakeStore(rows)
    embedder = FakeEmbedder()
    settings = RetrieveSettings()

    result = await retrieve_candidates(
        sq, typing.cast(VectorStore, store), typing.cast(Embedder, embedder), settings
    )

    assert [sc.chunk.position for sc in result] == [0, 1, 2]
    assert [sc.score for sc in result] == [-0.5, -0.1, -0.9]


async def test_retrieve_candidates_empty_store_returns_empty_list() -> None:
    sq = make_sub_question()
    store = FakeStore([])
    embedder = FakeEmbedder()
    settings = RetrieveSettings()

    result = await retrieve_candidates(
        sq, typing.cast(VectorStore, store), typing.cast(Embedder, embedder), settings
    )

    assert result == []


async def test_rerank_candidates_empty_short_circuits_without_touching_reranker() -> None:
    sq = make_sub_question()
    reranker = FakeReranker()
    settings = RetrieveSettings()

    result = await rerank_candidates(sq, [], typing.cast(Reranker, reranker), settings)

    assert result == []
    assert reranker.calls == []


async def test_rerank_candidates_passes_rerank_top_k() -> None:
    sq = make_sub_question()
    candidates = [ScoredChunk(chunk=make_chunk(0), score=1.0)]
    reranker = FakeReranker(result=candidates)
    settings = RetrieveSettings(ann_top_k=200, rerank_top_k=7)

    await rerank_candidates(sq, candidates, typing.cast(Reranker, reranker), settings)

    assert reranker.calls[0][2] == 7


async def test_rerank_candidates_passes_question_text_and_bare_chunks() -> None:
    sq = make_sub_question(question="How is heat discharged?")
    candidates = [
        ScoredChunk(chunk=make_chunk(0), score=1.0),
        ScoredChunk(chunk=make_chunk(1), score=0.5),
    ]
    reranker = FakeReranker(result=[])
    settings = RetrieveSettings()

    await rerank_candidates(sq, candidates, typing.cast(Reranker, reranker), settings)

    query, chunks, _top_k = reranker.calls[0]
    assert query == "How is heat discharged?"
    assert chunks == [c.chunk for c in candidates]


async def test_rerank_candidates_returns_reranker_output_unmodified() -> None:
    sq = make_sub_question()
    candidates = [ScoredChunk(chunk=make_chunk(0), score=1.0)]
    expected = [ScoredChunk(chunk=make_chunk(0), score=9.9)]
    reranker = FakeReranker(result=expected)
    settings = RetrieveSettings()

    result = await rerank_candidates(sq, candidates, typing.cast(Reranker, reranker), settings)

    assert result == expected


async def test_sub_question_requires_two_to_four_queries() -> None:
    with pytest.raises(ValueError, match="queries"):
        SubQuestion(id="sq02", question="q", queries=["only one"], success_criterion="x")
