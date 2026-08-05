"""Chart generation: SVG files exist, carry their labels, skip when empty."""

from __future__ import annotations

from pathlib import Path

from quarry_ldr.report.charts import (
    compression_funnel_chart,
    cost_by_stage_chart,
    evidence_per_subquestion_chart,
    render_charts,
    source_mix_chart,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_funnel_chart_is_svg_with_labels(tmp_path: Path) -> None:
    out = compression_funnel_chart(
        {"n_chunks": 500, "n_chunks_after_dedup": 400, "n_chunks_evidence": 120},
        tmp_path / "funnel.svg",
    )
    svg = _read(out)
    assert svg.lstrip().startswith("<?xml") or "<svg" in svg
    assert "Evidence compression" in svg


def test_source_mix_caps_domains(tmp_path: Path) -> None:
    urls = [f"https://site{i}.example/page" for i in range(12) for _ in range(i + 1)]
    out = source_mix_chart(urls, tmp_path / "sources.svg")
    svg = _read(out)
    assert "Source mix" in svg
    assert "site11.example" in svg  # the most frequent domain survives the cap
    assert "site0.example" not in svg  # the rarest does not


def test_evidence_chart_orders_sub_questions(tmp_path: Path) -> None:
    out = evidence_per_subquestion_chart(
        {"sq02": 5, "sq01": 9, "sq03": 2}, tmp_path / "coverage.svg"
    )
    svg = _read(out)
    assert "Coverage per sub-question" in svg
    assert "sq01" in svg and "sq03" in svg


def test_cost_chart_skips_zero_spend(tmp_path: Path) -> None:
    assert cost_by_stage_chart({"plan": 0.0, "gap": 0.0}, tmp_path / "cost.svg") is None
    assert not (tmp_path / "cost.svg").exists()
    out = cost_by_stage_chart({"plan": 0.13, "synthesize": 2.63}, tmp_path / "cost.svg")
    assert out is not None
    assert "API spend by stage" in _read(out)


def test_render_charts_builds_what_applies(tmp_path: Path) -> None:
    charts = render_charts(
        urls=["https://a.example/x", "https://b.example/y"],
        counts_by_sq={"sq01": 2},
        funnel={"n_chunks": 10, "n_chunks_after_dedup": 8, "n_chunks_evidence": 4},
        cost_by_stage={},
        out_dir=tmp_path,
    )
    assert set(charts) == {"funnel", "sources", "coverage"}  # no cost chart at $0
    assert all(path.is_file() for path in charts.values())


def test_render_charts_all_empty_is_empty(tmp_path: Path) -> None:
    assert render_charts([], {}, {}, {}, tmp_path) == {}
