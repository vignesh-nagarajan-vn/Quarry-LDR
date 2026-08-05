"""Real cross-encoder score distributions: supported vs unsupported claims.

Calibrates verify.floor. Run with
``uv run python -m pytest -m gpu tests/integration/test_verify_calibration.py -s --no-cov``
and read the printed quantiles; the shipped default must sit between the two
distributions with margin. Measured numbers live in DECISIONS.md.

Scores are raw logits (Reranker pins identity activation). Grade semantics
map onto entailment as follows: grade-3 chunks directly answer their query,
so (query, grade-3 chunk) is the weakest pair a genuinely supported claim
produces; grade-0 chunks are unrelated, the citation-points-at-garbage case
VERIFY exists to catch. Grades 1 and 2 are related-but-not-answering pairs,
a marginal band the floor is not asked to separate. The paraphrase anchors
below bound the real use case: a report sentence written from its cited
chunk scores far above any of this.

Skips when the reranker is not in the HF cache (same contract as the other
gpu-marked tests: fetch models outside pytest first).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from quarry_ldr.config import load_config
from quarry_ldr.gpu.arbiter import TorchCudaBackend, VramArbiter
from quarry_ldr.gpu.reranker import Reranker

pytestmark = pytest.mark.gpu

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _quantile(sorted_values: list[float], fraction: float) -> float:
    index = min(len(sorted_values) - 1, int(fraction * len(sorted_values)))
    return sorted_values[index]


async def test_supported_and_unsupported_scores_separate() -> None:
    data = json.loads((FIXTURES / "relevance.json").read_text("utf-8"))
    supported: list[tuple[str, str]] = []
    unsupported: list[tuple[str, str]] = []
    anchors: list[tuple[str, str]] = []  # (sentence, its own source chunk)
    cross: list[tuple[str, str]] = []  # (sentence, an unrelated chunk)
    for entry in data["queries"]:
        for chunk in entry["chunks"]:
            pair = (entry["query"], chunk["text"])
            if chunk["grade"] == 3:
                supported.append(pair)
            elif chunk["grade"] == 0:
                unsupported.append(pair)
        best = max(entry["chunks"], key=lambda c: int(c["grade"]))
        worst = min(entry["chunks"], key=lambda c: int(c["grade"]))
        if best["grade"] == 3 and worst["grade"] == 0:
            anchors.append((best["text"], best["text"]))
            cross.append((best["text"], worst["text"]))
    assert supported and unsupported and anchors and cross

    cfg = load_config()
    arbiter = VramArbiter(cfg.gpu.vram_budget_mb, TorchCudaBackend())
    reranker = Reranker(cfg, arbiter)
    try:
        supported_scores = sorted(await reranker.score_pairs(supported))
        unsupported_scores = sorted(await reranker.score_pairs(unsupported))
        anchor_scores = sorted(await reranker.score_pairs(anchors))
        cross_scores = sorted(await reranker.score_pairs(cross))
    finally:
        await arbiter.evict_all()

    print(
        f"\nsupported   n={len(supported_scores)} min={supported_scores[0]:.2f} "
        f"p10={_quantile(supported_scores, 0.10):.2f} "
        f"median={_quantile(supported_scores, 0.50):.2f} max={supported_scores[-1]:.2f}"
    )
    print(
        f"unsupported n={len(unsupported_scores)} min={unsupported_scores[0]:.2f} "
        f"median={_quantile(unsupported_scores, 0.50):.2f} "
        f"p90={_quantile(unsupported_scores, 0.90):.2f} max={unsupported_scores[-1]:.2f}"
    )
    print(
        f"anchors     self min={anchor_scores[0]:.2f} max={anchor_scores[-1]:.2f}   "
        f"cross min={cross_scores[0]:.2f} max={cross_scores[-1]:.2f}"
    )
    floor = cfg.verify.floor
    keeps = sum(score >= floor for score in supported_scores)
    rejects = sum(score < floor for score in unsupported_scores)
    print(
        f"shipped floor={floor}: keeps {keeps}/{len(supported_scores)} supported, "
        f"rejects {rejects}/{len(unsupported_scores)} unsupported"
    )

    # The distributions must separate decisively; the exact floor is a
    # judgment call recorded in DECISIONS.md from these printed numbers.
    assert statistics.mean(supported_scores) > statistics.mean(unsupported_scores) + 2.0
    # The shipped floor must not reject genuinely supported claims...
    assert keeps >= int(0.8 * len(supported_scores))
    # ...and must catch the unrelated-citation case it exists for.
    assert rejects >= int(0.8 * len(unsupported_scores))
    # The paraphrase anchors bound the real verify distribution: a sentence
    # scored against its own source must clear the floor with a wide margin,
    # and against an unrelated chunk must fall far below it.
    assert min(anchor_scores) > floor + 2.0
    assert max(cross_scores) < floor - 2.0
