"""Cross-encoder reranker: CPU-only tests against a fake model swapped in
through the arbiter's registration hook (never a real CrossEncoder)."""

from __future__ import annotations

import pytest

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.arbiter import ModelSpec, VramArbiter
from quarry_ldr.gpu.reranker import Reranker, ScoredChunk
from quarry_ldr.ingest.chunk import Chunk


def make_chunk(position: int, text: str, url: str = "https://example.com/doc") -> Chunk:
    return Chunk(
        chunk_id=f"chunk-{position}",
        url=url,
        text=text,
        token_count=max(1, len(text) // 4),
        position=position,
        start_char=0,
        end_char=len(text),
    )


class FakeCrossEncoder:
    """Scores (query, text) pairs by keyword overlap; records batch sizes seen."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.predict_calls = 0

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[float]:
        self.predict_calls += 1
        self.batch_sizes.append(batch_size)
        scores = []
        for query, text in pairs:
            q_words = set(query.lower().split())
            t_words = set(text.lower().split())
            scores.append(float(len(q_words & t_words)))
        return scores


def make_reranker(cfg: QuarryConfig, arbiter: VramArbiter) -> tuple[Reranker, FakeCrossEncoder]:
    """Construct a real Reranker, then swap its registered spec for a fake
    loader per the required CPU-test pattern (register only while not resident)."""
    reranker = Reranker(cfg, arbiter)
    fake = FakeCrossEncoder()
    arbiter.register(
        ModelSpec(
            name=Reranker.ARBITER_NAME,
            footprint_mb=10,
            loader=lambda: fake,
            unloader=lambda model: None,
        )
    )
    return reranker, fake


@pytest.fixture()
def arbiter() -> VramArbiter:
    return VramArbiter(budget_mb=4096, backend=None)


async def test_empty_chunks_returns_empty_without_acquiring(
    cfg: QuarryConfig, arbiter: VramArbiter
) -> None:
    reranker, fake = make_reranker(cfg, arbiter)
    result = await reranker.rerank("sand battery", [], top_k=5)
    assert result == []
    assert fake.predict_calls == 0
    assert arbiter.resident_models() == []


async def test_orders_by_score_desc(cfg: QuarryConfig, arbiter: VramArbiter) -> None:
    reranker, _fake = make_reranker(cfg, arbiter)
    chunks = [
        make_chunk(0, "cats and dogs are pets"),
        make_chunk(1, "sand battery thermal"),
        make_chunk(2, "thermal efficiency of the sand battery storage"),
    ]
    result = await reranker.rerank("sand battery thermal efficiency", chunks, top_k=3)
    scores = [sc.score for sc in result]
    assert scores == sorted(scores, reverse=True)
    assert result[0].chunk.position == 2  # highest keyword overlap


async def test_top_k_truncation(cfg: QuarryConfig, arbiter: VramArbiter) -> None:
    reranker, _fake = make_reranker(cfg, arbiter)
    chunks = [make_chunk(i, f"word{i} sand battery") for i in range(10)]
    result = await reranker.rerank("sand battery", chunks, top_k=4)
    assert len(result) == 4


async def test_ties_stable_by_input_order(cfg: QuarryConfig, arbiter: VramArbiter) -> None:
    reranker, _fake = make_reranker(cfg, arbiter)
    # None of these overlap the query at all: every score ties at 0.0.
    chunks = [make_chunk(i, "irrelevant filler text") for i in range(5)]
    result = await reranker.rerank("zzz nonexistent", chunks, top_k=5)
    assert [sc.chunk.position for sc in result] == [0, 1, 2, 3, 4]


async def test_scores_are_python_floats(cfg: QuarryConfig, arbiter: VramArbiter) -> None:
    reranker, _fake = make_reranker(cfg, arbiter)
    chunks = [make_chunk(0, "sand battery")]
    result = await reranker.rerank("sand battery", chunks, top_k=1)
    assert isinstance(result[0].score, float)


async def test_batch_size_passed_through_to_predict(
    cfg: QuarryConfig, arbiter: VramArbiter
) -> None:
    cfg.gpu.rerank_batch_size = 8
    reranker, fake = make_reranker(cfg, arbiter)
    chunks = [make_chunk(i, "sand battery") for i in range(3)]
    await reranker.rerank("sand battery", chunks, top_k=3)
    assert fake.batch_sizes == [8]


async def test_arbiter_holds_reranker_resident_after_call(
    cfg: QuarryConfig, arbiter: VramArbiter
) -> None:
    reranker, _fake = make_reranker(cfg, arbiter)
    await reranker.rerank("sand battery", [make_chunk(0, "sand battery")], top_k=1)
    assert arbiter.resident_models() == ["reranker"]


def test_init_registers_configured_footprint(cfg: QuarryConfig, arbiter: VramArbiter) -> None:
    cfg.gpu.footprints_mb["reranker"] = 999
    Reranker(cfg, arbiter)
    assert arbiter._specs["reranker"].footprint_mb == 999


def test_init_falls_back_to_default_footprint(cfg: QuarryConfig, arbiter: VramArbiter) -> None:
    cfg.gpu.footprints_mb.pop("reranker", None)
    Reranker(cfg, arbiter)
    assert arbiter._specs["reranker"].footprint_mb == 2128


async def test_rerank_returns_scored_chunk_instances(
    cfg: QuarryConfig, arbiter: VramArbiter
) -> None:
    reranker, _fake = make_reranker(cfg, arbiter)
    result = await reranker.rerank("sand battery", [make_chunk(0, "sand battery")], top_k=1)
    assert all(isinstance(sc, ScoredChunk) for sc in result)


async def test_all_chunks_scored_even_when_top_k_smaller(
    cfg: QuarryConfig, arbiter: VramArbiter
) -> None:
    reranker, fake = make_reranker(cfg, arbiter)
    chunks = [make_chunk(i, "sand battery") for i in range(6)]
    await reranker.rerank("sand battery", chunks, top_k=2)
    # predict() must have been called once, over every candidate, not just top_k.
    assert fake.predict_calls == 1


async def test_score_pairs_empty_short_circuits(cfg: QuarryConfig, arbiter: VramArbiter) -> None:
    reranker, fake = make_reranker(cfg, arbiter)
    assert await reranker.score_pairs([]) == []
    assert fake.predict_calls == 0
    assert arbiter.resident_models() == []


async def test_score_pairs_returns_floats_and_passes_batch_size(
    cfg: QuarryConfig, arbiter: VramArbiter
) -> None:
    cfg.gpu.rerank_batch_size = 8
    reranker, fake = make_reranker(cfg, arbiter)
    scores = await reranker.score_pairs(
        [("sand battery", "sand battery text"), ("sand battery", "cats")]
    )
    assert all(isinstance(score, float) for score in scores)
    assert scores[0] > scores[1]
    assert fake.batch_sizes == [8]
