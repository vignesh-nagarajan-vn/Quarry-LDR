"""VERIFY stage: every cited sentence is scored against its cited chunks.

The cross-encoder that already ranks retrieval candidates doubles as an
entailment judge: (sentence, cited evidence) pairs are scored in one batched
pass, and only sentences under ``verify.floor`` pay for rewrite attempts on
the triage model (which fits alongside the reranker under the VRAM budget,
so the whole stage costs at most one llama-server swap). A sentence no
rewrite can lift above the floor is dropped; the references list is a
superset by construction, so removing markers never breaks
validate_markdown. Runs in every engine mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.local_llm import LlamaServer, LocalLLM
from quarry_ldr.gpu.reranker import Reranker
from quarry_ldr.ingest.chunk import Chunk
from quarry_ldr.logging import get_logger
from quarry_ldr.pipeline.synthesize import DraftReport, ReportSection
from quarry_ldr.report.citations import CITATION_MARKER, CitationError, CitationIndex

logger = get_logger(component="verify")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LIST_LINE = re.compile(r"\s*(?:[-*]|\d+[.)])\s+")
_EVIDENCE_CHARS_PER_CHUNK = 1500
_EVIDENCE_CHARS_TOTAL = 6000
_REWRITE_MAX_TOKENS = 256


class ClaimVerdict(BaseModel):
    section_title: str
    sentence: str
    citation_numbers: list[int] = Field(default_factory=list)
    entailment_score: float
    action: Literal["kept", "rewritten", "dropped"]
    rewrite_attempts: int = 0


class VerificationSummary(BaseModel):
    total_claims: int = 0
    kept: int = 0
    rewritten: int = 0
    dropped: int = 0
    mean_entailment_score: float = 0.0
    verdicts: list[ClaimVerdict] = Field(default_factory=list)


class _RewritePayload(BaseModel):
    sentence: str


REWRITE_PROMPT = """The sentence below claims more than its cited evidence supports.

Evidence:
---
{evidence}
---

Sentence: {sentence}

Rewrite the sentence so every claim is directly supported by the evidence
above. Keep the citation markers {markers} exactly as they are; add nothing
the evidence does not say. Reply with JSON only: {{"sentence": "..."}}"""


@dataclass
class _Unit:
    """One verifiable sentence: it carries at least one resolving citation."""

    text: str
    numbers: list[int]
    evidence: str
    score: float = 0.0
    action: Literal["kept", "rewritten", "dropped"] = "kept"
    rewrite_attempts: int = 0


@dataclass
class _Sentence:
    """One sentence (or list line) in a prose block; unit is None when there
    is nothing to verify and the text passes through untouched."""

    text: str
    unit: _Unit | None = None


@dataclass
class _Block:
    """A blank-line-separated markdown block."""

    passthrough: str | None = None  # headings/tables/quotes/code, verbatim
    is_list: bool = False
    sentences: list[_Sentence] = field(default_factory=list)


def _evidence_for(
    numbers: list[int], citations: CitationIndex, chunk_lookup: dict[str, Chunk]
) -> str:
    """Concatenated cited chunk texts; empty when nothing resolves.

    VERIFY runs before render's validate_markdown, so a marker that does not
    resolve is skipped defensively here and judged there.
    """
    parts: list[str] = []
    for number in numbers:
        try:
            citation = citations.get(number)
        except CitationError:
            continue
        chunk = chunk_lookup.get(citation.chunk_id)
        if chunk is not None:
            parts.append(chunk.text[:_EVIDENCE_CHARS_PER_CHUNK])
    return "\n\n".join(parts)[:_EVIDENCE_CHARS_TOTAL]


def _parse_block(block: str, citations: CitationIndex, chunk_lookup: dict[str, Chunk]) -> _Block:
    stripped = block.strip()
    if not stripped or stripped.startswith(("#", ">", "|", "```")):
        return _Block(passthrough=stripped)
    lines = [line for line in stripped.splitlines() if line.strip()]
    is_list = all(_LIST_LINE.match(line) for line in lines)
    pieces = lines if is_list else _SENTENCE_SPLIT.split(stripped)
    sentences: list[_Sentence] = []
    for piece in pieces:
        numbers = [int(m.group(1)) for m in CITATION_MARKER.finditer(piece)]
        evidence = _evidence_for(numbers, citations, chunk_lookup) if numbers else ""
        unit = _Unit(text=piece, numbers=numbers, evidence=evidence) if evidence else None
        sentences.append(_Sentence(text=piece, unit=unit))
    return _Block(is_list=is_list, sentences=sentences)


async def _rewrite_or_drop(
    unit: _Unit, reranker: Reranker, llm: LocalLLM, cfg: QuarryConfig
) -> None:
    """Up to verify.max_rewrites attempts on the triage model; every accepted
    rewrite must keep a non-empty subset of the original markers and re-score
    above the floor, else the sentence is dropped."""
    original_numbers = set(unit.numbers)
    markers = " ".join(f"[{n}]" for n in unit.numbers)
    for attempt in range(1, cfg.verify.max_rewrites + 1):
        unit.rewrite_attempts = attempt
        prompt = REWRITE_PROMPT.format(evidence=unit.evidence, sentence=unit.text, markers=markers)
        try:
            payload = await llm.complete_typed(
                prompt,
                _RewritePayload,
                max_tokens=_REWRITE_MAX_TOKENS,
                max_retries=cfg.triage.max_retries,
            )
        except ValueError:
            continue
        candidate = payload.sentence.strip()
        candidate_numbers = {int(m.group(1)) for m in CITATION_MARKER.finditer(candidate)}
        if not candidate or not candidate_numbers or not candidate_numbers <= original_numbers:
            continue
        (score,) = await reranker.score_pairs([(candidate, unit.evidence)])
        if score >= cfg.verify.floor:
            unit.text = candidate
            unit.score = score
            unit.action = "rewritten"
            return
    unit.action = "dropped"


async def verify_report(
    draft: DraftReport,
    citations: CitationIndex,
    chunk_lookup: dict[str, Chunk],
    reranker: Reranker,
    llm: LocalLLM,
    llama_server: LlamaServer | None,
    cfg: QuarryConfig,
) -> tuple[DraftReport, VerificationSummary]:
    """Two batched GPU passes: score everything, then rewrite-or-drop only
    the failures. Returns the cleaned draft plus the audit summary."""
    parsed: list[tuple[ReportSection, list[_Block]]] = []
    units: list[tuple[str, _Unit]] = []
    for section in draft.sections:
        blocks = [
            _parse_block(block, citations, chunk_lookup)
            for block in re.split(r"\n\s*\n", section.markdown)
        ]
        parsed.append((section, blocks))
        for block in blocks:
            for sentence in block.sentences:
                if sentence.unit is not None:
                    units.append((section.title, sentence.unit))

    if not units:
        return draft, VerificationSummary()

    scores = await reranker.score_pairs([(unit.text, unit.evidence) for _, unit in units])
    for (_, unit), score in zip(units, scores, strict=True):
        unit.score = score

    failing = [unit for _, unit in units if unit.score < cfg.verify.floor]
    if failing and llama_server is not None:
        # The reranker and the triage model fit together under the budget,
        # so this is the stage's only potential llama-server swap.
        await llama_server.start()
    for unit in failing:
        await _rewrite_or_drop(unit, reranker, llm, cfg)

    new_sections: list[ReportSection] = []
    for section, blocks in parsed:
        out_blocks: list[str] = []
        for block in blocks:
            if block.passthrough is not None:
                if block.passthrough:
                    out_blocks.append(block.passthrough)
                continue
            kept = [
                sentence.text if sentence.unit is None else sentence.unit.text
                for sentence in block.sentences
                if sentence.unit is None or sentence.unit.action != "dropped"
            ]
            if not kept:
                continue
            out_blocks.append(("\n" if block.is_list else " ").join(kept))
        new_sections.append(
            ReportSection(title=section.title, markdown="\n\n".join(out_blocks).strip())
        )

    survivor_scores = [unit.score for _, unit in units if unit.action != "dropped"]
    summary = VerificationSummary(
        total_claims=len(units),
        kept=sum(1 for _, unit in units if unit.action == "kept"),
        rewritten=sum(1 for _, unit in units if unit.action == "rewritten"),
        dropped=sum(1 for _, unit in units if unit.action == "dropped"),
        mean_entailment_score=(
            sum(survivor_scores) / len(survivor_scores) if survivor_scores else 0.0
        ),
        verdicts=[
            ClaimVerdict(
                section_title=title,
                sentence=unit.text,
                citation_numbers=unit.numbers,
                entailment_score=unit.score,
                action=unit.action,
                rewrite_attempts=unit.rewrite_attempts,
            )
            for title, unit in units
        ],
    )
    logger.info(
        "verify_scored",
        total=summary.total_claims,
        kept=summary.kept,
        rewritten=summary.rewritten,
        dropped=summary.dropped,
    )
    verified = DraftReport(topic=draft.topic, sections=new_sections, corpus_hash=draft.corpus_hash)
    return verified, summary
