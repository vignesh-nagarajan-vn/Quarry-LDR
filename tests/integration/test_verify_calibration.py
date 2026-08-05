"""Real cross-encoder score distributions: supported vs unsupported claims.

Calibrates verify.floor. Run with
``uv run python -m pytest -m gpu tests/integration/test_verify_calibration.py -s --no-cov``
and read the printed quantiles; the shipped default must sit between the two
distributions with margin. Measured numbers live in DECISIONS.md.

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
    for entry in data["queries"]:
        for chunk in entry["chunks"]:
            pair = (entry["query"], chunk["text"])
            if chunk["grade"] >= 2:
                supported.append(pair)
            elif chunk["grade"] == 0:
                unsupported.append(pair)
    assert supported and unsupported

    cfg = load_config()
    arbiter = VramArbiter(cfg.gpu.vram_budget_mb, TorchCudaBackend())
    reranker = Reranker(cfg, arbiter)
    try:
        supported_scores = sorted(await reranker.score_pairs(supported))
        unsupported_scores = sorted(await reranker.score_pairs(unsupported))
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
    # The shipped floor must not reject the bulk of genuinely supported claims.
    assert keeps >= int(0.8 * len(supported_scores))
