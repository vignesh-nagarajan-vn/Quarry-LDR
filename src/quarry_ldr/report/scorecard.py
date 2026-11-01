"""Report scorecard: measures a finished report from its markdown alone.

Deterministic and offline, so it scores old and new reports alike and can run
in tests. Thresholds live in ``evals/thresholds.json``; a threshold changes in
its own commit with a reason, never to make a report pass.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from quarry_ldr.ingest.sources import STRONG_TIERS, SourceTier, rate_url, site_of
from quarry_ldr.report.citations import CITATION_MARKER
from quarry_ldr.report.claims import (
    extract_figures,
    is_factual,
    named_entities,
    split_sentences,
    strip_markers,
)

REQUIRED_SECTIONS = ("Key findings", "What we could not establish", "Sources")
_NON_PROSE_SECTIONS = frozenset({"Sources"})
_QUESTION_START = re.compile(
    r"^(what|how|why|which|when|who|where|is|are|do|does|can|should|will)\b", re.IGNORECASE
)
_DANGLING_END = re.compile(r"\b(of|in|the|and|for|to|on|with|a|an|by|at|from|or)$", re.IGNORECASE)
MALFORMED_MARKER = re.compile(
    r"\[\d+\s*\([^\]]*\)\]|\[\d+\s*[,;]\s*\d+[^\]]*\]|\[\d+\s*[-\u2013]\s*\d+\]"
)
_REFERENCE = re.compile(r"(?m)^\[(\d+)\] .*? - (https?://\S+)")
_TOTAL_COST = re.compile(r"\*\*Total: \$([\d.]+)\*\*")
_MANIFEST = re.compile(r"## Run manifest\s+```json\s+(\{.*?\})\s+```", re.DOTALL)
_DEFAULT_THRESHOLDS = Path(__file__).resolve().parents[3] / "evals" / "thresholds.json"


class Scorecard(BaseModel):
    words: int
    sections: int
    citations: int
    distinct_sites: int
    top_site_share: float
    top3_site_share: float
    strong_source_share: float
    low_source_share: float
    uncited_factual_share: float
    malformed_markers: int
    cut_off_titles: int
    question_titles: int
    question_sentences: int
    longest_paragraph_sentences: int
    repeated_sentences: int
    specifics_per_1k_words: float
    missing_required_sections: int
    cost_usd: float
    minutes: float


class ThresholdResult(BaseModel):
    metric: str
    value: float
    passed: bool
    warn_only: bool = False
    limit: str


def _split_report(markdown: str) -> tuple[list[tuple[str, str]], str]:
    """Report sections as (title, body), plus everything from References on."""
    body, _, tail = markdown.partition("## References")
    parts = re.split(r"(?m)^## (.+)$", body)
    return list(zip(parts[1::2], (b.strip() for b in parts[2::2]), strict=True)), tail


def _prose_paragraphs(body: str) -> list[str]:
    """Blank-line blocks that hold prose, not headings, tables, or quotes."""
    return [
        block.strip()
        for block in re.split(r"\n\s*\n", body)
        if block.strip() and not block.lstrip().startswith(("#", "|", ">", "```"))
    ]


def _block_sentences(block: str) -> list[str]:
    lines = [line for line in block.splitlines() if line.strip()]
    if all(re.match(r"\s*(?:[-*]|\d+[.)])\s+", line) for line in lines):
        return lines
    return split_sentences(" ".join(lines))


def _share(count: int, total: int) -> float:
    return round(count / total, 3) if total else 0.0


def score_report(markdown: str, overrides: dict[str, str] | None = None) -> Scorecard:
    sections, tail = _split_report(markdown)
    prose_sections = [(t, b) for t, b in sections if t.strip() not in _NON_PROSE_SECTIONS]
    paragraphs = [p for _, body in prose_sections for p in _prose_paragraphs(body)]
    per_paragraph = [_block_sentences(p) for p in paragraphs]
    sentences = [s for group in per_paragraph for s in group]
    prose = "\n\n".join(paragraphs)

    refs = dict(_REFERENCE.findall(tail))
    cited_urls = [refs[n] for n in CITATION_MARKER.findall(prose) if n in refs]
    by_site = Counter(site_of(url) for url in cited_urls)
    tiers = Counter(rate_url(url, overrides) for url in cited_urls)
    n_cited = len(cited_urls)

    uncited = [s for s in sentences if not CITATION_MARKER.search(s) and is_factual(s)]
    counts = Counter(strip_markers(s).lower() for s in sentences if len(s.split()) >= 6)
    words = len(strip_markers(prose).split())
    titles = [title.strip() for title, _ in sections]

    manifest_match = _MANIFEST.search(markdown)
    minutes = 0.0
    if manifest_match:
        manifest = json.loads(manifest_match.group(1))
        started = datetime.fromisoformat(manifest["started_at"])
        finished = datetime.fromisoformat(manifest["finished_at"])
        minutes = round((finished - started).total_seconds() / 60, 1)
    cost_match = _TOTAL_COST.search(markdown)

    entities = named_entities(prose)
    return Scorecard(
        words=words,
        sections=len(sections),
        citations=n_cited,
        distinct_sites=len(by_site),
        top_site_share=_share(by_site.most_common(1)[0][1], n_cited) if by_site else 0.0,
        top3_site_share=_share(sum(c for _, c in by_site.most_common(3)), n_cited),
        strong_source_share=_share(sum(tiers[t] for t in STRONG_TIERS), n_cited),
        low_source_share=_share(tiers[SourceTier.LOW], n_cited),
        uncited_factual_share=round(len(uncited) / len(sentences), 3) if sentences else 0.0,
        malformed_markers=len(MALFORMED_MARKER.findall(prose)),
        cut_off_titles=sum(
            1
            for t in titles
            if _DANGLING_END.search(t) or (_QUESTION_START.match(t) and not t.endswith("?"))
        ),
        question_titles=sum(1 for t in titles if t.endswith("?")),
        question_sentences=sum(1 for s in sentences if strip_markers(s).endswith("?")),
        longest_paragraph_sentences=max((len(g) for g in per_paragraph), default=0),
        repeated_sentences=sum(c - 1 for c in counts.values() if c > 1),
        specifics_per_1k_words=(
            round((len(entities) + len(extract_figures(prose))) * 1000 / words, 1) if words else 0.0
        ),
        missing_required_sections=sum(1 for req in REQUIRED_SECTIONS if req not in titles),
        cost_usd=float(cost_match.group(1)) if cost_match else 0.0,
        minutes=minutes,
    )


def load_thresholds(path: Path | None = None) -> dict[str, dict[str, float | bool]]:
    loaded: dict[str, dict[str, float | bool]] = json.loads(
        (path or _DEFAULT_THRESHOLDS).read_text(encoding="utf-8")
    )
    return loaded


def evaluate(
    card: Scorecard, thresholds: dict[str, dict[str, float | bool]]
) -> list[ThresholdResult]:
    """Compare a scorecard against thresholds. An unknown metric name raises,
    so a renamed metric cannot silently stop being gated."""
    values = card.model_dump()
    results: list[ThresholdResult] = []
    for metric, limits in thresholds.items():
        value = float(values[metric])
        low, high = limits.get("min"), limits.get("max")
        passed = (low is None or value >= float(low)) and (high is None or value <= float(high))
        bounds = [f">= {low}" if low is not None else "", f"<= {high}" if high is not None else ""]
        results.append(
            ThresholdResult(
                metric=metric,
                value=value,
                passed=passed,
                warn_only=bool(limits.get("warn_only", False)),
                limit=" and ".join(b for b in bounds if b),
            )
        )
    return results
