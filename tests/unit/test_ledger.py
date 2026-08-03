"""Ledger arithmetic against hand-computed fixtures, including cache reads,
batch discounts, date-aware Sonnet pricing, and the cost cap."""

from __future__ import annotations

from datetime import date

import pytest

from quarry_ldr.ledger import (
    BATCH_DISCOUNT,
    CostCapExceeded,
    Ledger,
    TokenUsage,
    UnknownModelError,
    compute_cost,
    price_for,
)

RUN_DATE = date(2026, 8, 3)


def test_plain_opus_call_hand_computed() -> None:
    # 60K input at $5/MTok = $0.30; 2K output at $25/MTok = $0.05.
    usage = TokenUsage(input_tokens=60_000, output_tokens=2_000)
    cost = compute_cost(usage, price_for("claude-opus-5", RUN_DATE))
    assert cost == pytest.approx(0.35)


def test_cached_synthesis_matches_spec_math() -> None:
    """The spec's worked example: 10 section calls over a 60K corpus on Opus.

    1h cache: 60K write at $10/MTok ($0.60) + 9 reads x 60K at $0.50/MTok
    ($0.27) = $0.87, versus $3.00 uncached.
    """
    ledger = Ledger()
    price_date = RUN_DATE
    ledger.record(
        "claude-opus-5",
        TokenUsage(cache_creation_input_tokens=60_000),
        stage="synthesize",
        on=price_date,
    )
    for _ in range(9):
        ledger.record(
            "claude-opus-5",
            TokenUsage(cache_read_input_tokens=60_000),
            stage="synthesize",
            on=price_date,
        )
    assert ledger.total_cost_usd == pytest.approx(0.87)

    uncached = compute_cost(
        TokenUsage(input_tokens=600_000), price_for("claude-opus-5", price_date)
    )
    assert uncached == pytest.approx(3.00)


def test_sonnet_pricing_is_date_aware() -> None:
    usage = TokenUsage(input_tokens=1_000_000)
    before = compute_cost(usage, price_for("claude-sonnet-5", date(2026, 8, 31)))
    after = compute_cost(usage, price_for("claude-sonnet-5", date(2026, 9, 1)))
    assert before == pytest.approx(2.0)
    assert after == pytest.approx(3.0)
    out = TokenUsage(output_tokens=1_000_000)
    assert compute_cost(out, price_for("claude-sonnet-5", date(2026, 9, 1))) == pytest.approx(15.0)


def test_batch_halves_everything_and_stacks_with_cache() -> None:
    price = price_for("claude-haiku-4-5-20251001", RUN_DATE)
    # Batch input: 1M at $1 -> $0.50.
    assert compute_cost(TokenUsage(input_tokens=1_000_000), price, batch=True) == pytest.approx(0.5)
    # Batch + cache read stack: 1M cache reads at $0.10 -> $0.05.
    assert compute_cost(
        TokenUsage(cache_read_input_tokens=1_000_000), price, batch=True
    ) == pytest.approx(0.05)
    assert BATCH_DISCOUNT == 0.5


def test_cost_cap_raises_but_still_records() -> None:
    ledger = Ledger(cost_cap_usd=0.50)
    with pytest.raises(CostCapExceeded) as excinfo:
        ledger.record(
            "claude-opus-5",
            TokenUsage(input_tokens=100_000, output_tokens=4_000),  # $0.50 + $0.10
            stage="synthesize",
            on=RUN_DATE,
        )
    assert excinfo.value.cap_usd == 0.50
    assert len(ledger.entries) == 1
    assert ledger.total_cost_usd == pytest.approx(0.60)


def test_unknown_model_raises() -> None:
    with pytest.raises(UnknownModelError, match="claude-nonexistent"):
        price_for("claude-nonexistent", RUN_DATE)


def test_date_before_first_entry_uses_earliest() -> None:
    price = price_for("claude-sonnet-5", date(2024, 1, 1))
    assert price.input == 2.0


def test_summary_groups_by_model_stage_iteration() -> None:
    ledger = Ledger()
    ledger.record("claude-opus-5", TokenUsage(input_tokens=10_000), stage="plan", on=RUN_DATE)
    ledger.record(
        "claude-sonnet-5", TokenUsage(input_tokens=10_000), stage="gap", iteration=1, on=RUN_DATE
    )
    ledger.record(
        "claude-sonnet-5", TokenUsage(input_tokens=10_000), stage="gap", iteration=2, on=RUN_DATE
    )
    summary = ledger.summary()
    assert summary.total_input_tokens == 30_000
    assert summary.by_model["claude-opus-5"] == pytest.approx(0.05)
    assert summary.by_model["claude-sonnet-5"] == pytest.approx(0.04)
    assert summary.by_stage["gap"] == pytest.approx(0.04)
    assert summary.by_iteration[1] == pytest.approx(0.02)


def test_dump_load_round_trip() -> None:
    ledger = Ledger(cost_cap_usd=5.0)
    ledger.record(
        "claude-opus-5",
        TokenUsage(input_tokens=1000, output_tokens=200, cache_read_input_tokens=500),
        stage="plan",
        context="unit test",
        on=RUN_DATE,
    )
    restored = Ledger.load(ledger.dump(), cost_cap_usd=5.0)
    assert restored.total_cost_usd == pytest.approx(ledger.total_cost_usd)
    assert restored.entries[0].context == "unit test"
    assert restored.entries[0].usage.cache_read_input_tokens == 500


def test_to_markdown_contains_totals_and_rows() -> None:
    ledger = Ledger(cost_cap_usd=1.0)
    ledger.record("claude-opus-5", TokenUsage(input_tokens=10_000), stage="plan", on=RUN_DATE)
    ledger.record(
        "claude-sonnet-5", TokenUsage(input_tokens=5_000), stage="gap", iteration=1, on=RUN_DATE
    )
    md = ledger.to_markdown()
    assert "## Cost ledger" in md
    assert "claude-opus-5" in md
    assert "$0.0500" in md
    assert "Total: $0.0600" in md
    assert "cap $1.00" in md
    assert "Per iteration" in md
