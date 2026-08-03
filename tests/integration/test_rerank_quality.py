"""GPU integration: the real cross-encoder must beat a lexical baseline.

Loads tests/fixtures/relevance.json (5 queries x 6 graded chunks, grades
0-3), computes mean nDCG@6 for a word-overlap lexical ranking and for the
real BAAI/bge-reranker-v2-m3 CrossEncoder (via Reranker + a real VramArbiter
over TorchCudaBackend), and asserts the model beats the baseline.

Requires a CUDA GPU and downloads the reranker model on first run;
deselected by default (-m gpu).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.arbiter import TorchCudaBackend, VramArbiter
from quarry_ldr.gpu.reranker import Reranker
from quarry_ldr.ingest.chunk import Chunk
from quarry_ldr.logging import get_logger

pytestmark = pytest.mark.gpu

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
ARBITER_BUDGET_MB = 6656


def _make_chunk(position: int, text: str) -> Chunk:
    """Minimal Chunk wrapper: only the fields the reranker actually reads matter."""
    return Chunk(
        chunk_id=f"relevance-{position}",
        url="https://example.com/relevance-fixture",
        text=text,
        token_count=max(1, len(text) // 4),
        position=position,
        start_char=0,
        end_char=len(text),
    )


def _lexical_score(query: str, text: str) -> float:
    """Baseline ranker: fraction of query words present in the chunk text."""
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    if not q_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)


def _dcg(grades: list[int]) -> float:
    return sum(grade / math.log2(idx + 2) for idx, grade in enumerate(grades))


def _ndcg_at_k(ranked_grades: list[int], k: int) -> float:
    actual = _dcg(ranked_grades[:k])
    ideal = _dcg(sorted(ranked_grades, reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return actual / ideal


@pytest.fixture()
def relevance_data() -> dict[str, Any]:
    with (FIXTURES_DIR / "relevance.json").open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


async def test_reranker_ndcg_beats_lexical_baseline(
    cfg: QuarryConfig, relevance_data: dict[str, Any]
) -> None:
    arbiter = VramArbiter(budget_mb=ARBITER_BUDGET_MB, backend=TorchCudaBackend())
    reranker = Reranker(cfg, arbiter)
    log = get_logger(component="test_rerank_quality")

    baseline_scores: list[float] = []
    model_scores: list[float] = []

    try:
        for query_case in relevance_data["queries"]:
            query = query_case["query"]
            graded = query_case["chunks"]
            chunks = [_make_chunk(i, c["text"]) for i, c in enumerate(graded)]
            grade_by_id = {
                chunk.chunk_id: c["grade"] for chunk, c in zip(chunks, graded, strict=True)
            }

            baseline_order = sorted(
                chunks, key=lambda ch: _lexical_score(query, ch.text), reverse=True
            )
            baseline_grades = [grade_by_id[ch.chunk_id] for ch in baseline_order]
            baseline_scores.append(_ndcg_at_k(baseline_grades, 6))

            reranked = await reranker.rerank(query, chunks, top_k=6)
            model_grades = [grade_by_id[sc.chunk.chunk_id] for sc in reranked]
            model_scores.append(_ndcg_at_k(model_grades, 6))
    finally:
        await arbiter.evict_all()

    mean_baseline = sum(baseline_scores) / len(baseline_scores)
    mean_model = sum(model_scores) / len(model_scores)
    log.info("rerank_quality", mean_baseline_ndcg=mean_baseline, mean_model_ndcg=mean_model)

    assert mean_model >= mean_baseline
