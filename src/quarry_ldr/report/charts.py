"""Report charts: matplotlib SVGs for the PDF deliverable, CPU only.

The markdown report stays plain; charts render only into the PDF. The Agg
backend is forced before pyplot loads so headless machines and CI never need
a display. Every chart derives from run data already in hand, so a chart can
never claim anything the report does not.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ACCENT = "#b45309"  # amber-700, the Quarry accent
NEUTRAL = "#64748b"  # slate-500
DROPPED = "#b91c1c"  # red-700, only for verification losses
_MAX_DOMAINS = 8
_MAX_SQ_BARS = 15


def _style(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def _save(fig: Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def compression_funnel_chart(funnel: dict[str, int], out_path: Path) -> Path:
    """Raw chunks, after dedup, evidence: the compression story."""
    labels = ["chunks", "after dedup", "evidence"]
    values = [
        int(funnel.get("n_chunks", 0)),
        int(funnel.get("n_chunks_after_dedup", 0)),
        int(funnel.get("n_chunks_evidence", 0)),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    bars = ax.barh(labels[::-1], values[::-1], color=[ACCENT, NEUTRAL, NEUTRAL])
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel("chunks", fontsize=8)
    ax.set_title("Evidence compression", fontsize=10, loc="left")
    _style(ax)
    return _save(fig, out_path)


def source_mix_chart(urls: list[str], out_path: Path) -> Path:
    """Top evidence domains by chunk count."""
    domains = Counter(urlparse(url).netloc for url in urls if url)
    top = domains.most_common(_MAX_DOMAINS)
    labels = [domain for domain, _ in top][::-1]
    values = [count for _, count in top][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.55 + 0.34 * max(1, len(labels))))
    bars = ax.barh(labels, values, color=NEUTRAL)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel("evidence chunks", fontsize=8)
    ax.set_title("Source mix", fontsize=10, loc="left")
    _style(ax)
    return _save(fig, out_path)


def evidence_per_subquestion_chart(counts_by_sq: dict[str, int], out_path: Path) -> Path:
    """Evidence chunk count per sub-question, in plan order."""
    items = sorted(counts_by_sq.items())[:_MAX_SQ_BARS]
    labels = [sq_id for sq_id, _ in items]
    values = [count for _, count in items]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.bar(labels, values, color=NEUTRAL)
    ax.set_ylabel("evidence chunks", fontsize=8)
    ax.set_title("Coverage per sub-question", fontsize=10, loc="left")
    _style(ax)
    return _save(fig, out_path)


def cost_by_stage_chart(by_stage: dict[str, float], out_path: Path) -> Path | None:
    """API spend per stage; None when the run spent nothing (local mode)."""
    spending = {stage: cost for stage, cost in by_stage.items() if cost > 0}
    if not spending:
        return None
    items = sorted(spending.items(), key=lambda kv: -kv[1])
    labels = [stage for stage, _ in items][::-1]
    values = [cost for _, cost in items][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.55 + 0.34 * max(1, len(labels))))
    bars = ax.barh(labels, values, color=ACCENT)
    ax.bar_label(bars, padding=3, fontsize=8, fmt="$%.4f")
    ax.set_xlabel("USD", fontsize=8)
    ax.set_title("API spend by stage", fontsize=10, loc="left")
    _style(ax)
    return _save(fig, out_path)


def verification_chart(verification: dict[str, int], out_path: Path) -> Path | None:
    """Kept / rewritten / dropped claims as one stacked bar; None when VERIFY
    checked nothing (disabled, or a report with no cited sentences)."""
    kept = int(verification.get("n_claims_checked", 0)) - (
        int(verification.get("n_claims_rewritten", 0))
        + int(verification.get("n_claims_dropped", 0))
    )
    rewritten = int(verification.get("n_claims_rewritten", 0))
    dropped = int(verification.get("n_claims_dropped", 0))
    total = kept + rewritten + dropped
    if total <= 0:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 1.5))
    left = 0
    for label, value, color in (
        ("kept", kept, ACCENT),
        ("rewritten", rewritten, NEUTRAL),
        ("dropped", dropped, DROPPED),
    ):
        # Zero-count categories still get a (zero-width) bar so the legend
        # tells the whole story: "dropped (0)" is the claim, not an omission.
        ax.barh([""], [value], left=left, color=color, label=f"{label} ({value})")
        left += value
    ax.set_xlim(0, total)
    ax.set_xlabel("cited sentences", fontsize=8)
    ax.set_title("Claim verification", fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncols=3, fontsize=8, frameon=False)
    _style(ax)
    ax.set_yticks([])
    return _save(fig, out_path)


def render_charts(
    urls: list[str],
    counts_by_sq: dict[str, int],
    funnel: dict[str, int],
    cost_by_stage: dict[str, float],
    out_dir: Path,
    verification: dict[str, int] | None = None,
) -> dict[str, Path]:
    """Every chart that applies to this run; empty inputs skip their chart."""
    charts: dict[str, Path] = {}
    if any(int(funnel.get(key, 0)) for key in ("n_chunks", "n_chunks_after_dedup")):
        charts["funnel"] = compression_funnel_chart(funnel, out_dir / "funnel.svg")
    if urls:
        charts["sources"] = source_mix_chart(urls, out_dir / "sources.svg")
    if counts_by_sq:
        charts["coverage"] = evidence_per_subquestion_chart(counts_by_sq, out_dir / "coverage.svg")
    if verification is not None:
        verify_chart = verification_chart(verification, out_dir / "verification.svg")
        if verify_chart is not None:
            charts["verification"] = verify_chart
    cost_chart = cost_by_stage_chart(cost_by_stage, out_dir / "cost.svg")
    if cost_chart is not None:
        charts["cost"] = cost_chart
    return charts
