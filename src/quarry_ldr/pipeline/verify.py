"""VERIFY stage: every cited sentence is checked against its cited chunks.

Three gates, all of which a sentence must pass to stand as written:

1. Relevance: the cross-encoder scores (sentence, cited evidence) pairs in one
   batched pass against ``verify.floor``.
2. Figures: code checks that every number in the sentence appears in its
   cited evidence after normalization.
3. Support: the local triage model answers ``supported | partial |
   unsupported`` as schema-validated JSON; code passes only ``supported``. A
   sentence the model cannot judge after retries fails closed.

A failing sentence pays for rewrite attempts on the triage model, and a
rewrite stands only if it passes all three gates again; otherwise the
sentence is dropped. Uncited factual sentences get one chance: the reranker
picks the best chunk the section already cites, the support check confirms
it, and code attaches that marker; otherwise the sentence is dropped.
Linking sentences and labeled gap notes stay. The model sees the whole
cited chunk within the triage context budget. The reranker and the triage
model fit together under the VRAM budget, so the stage costs at most one
llama-server swap. Runs in every engine mode, always on the local GPU.
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
from quarry_ldr.pipeline.findings import GAPS_TITLE
from quarry_ldr.pipeline.synthesize import DraftReport, ReportSection
from quarry_ldr.report.citations import CITATION_MARKER, CitationError, CitationIndex
from quarry_ldr.report.claims import extract_figures, is_factual, split_sentences, strip_markers

logger = get_logger(component="verify")

_LIST_LINE = re.compile(r"\s*(?:[-*]|\d+[.)])\s+")
_TRAILING_PUNCT = re.compile(r"^(.*?)([.!?]+[\"')\]]*)?$", re.DOTALL)
_REWRITE_MAX_TOKENS = 256
_CHECK_MAX_TOKENS = 64
# Prompt scaffold plus the sentence, in tokens, kept free of evidence.
_PROMPT_OVERHEAD_TOKENS = 700
# Conservative characters per token for the evidence budget (English runs ~4).
_CHARS_PER_TOKEN = 3
# Code-built sections whose uncited sentences are gap statements by design.
_NO_ATTACH_SECTIONS = frozenset({GAPS_TITLE})

Action = Literal["kept", "rewritten", "dropped"]
Support = Literal["supported", "partial", "unsupported"]


class ClaimVerdict(BaseModel):
    section_title: str
    sentence: str
    citation_numbers: list[int] = Field(default_factory=list)
    entailment_score: float
    action: Action
    rewrite_attempts: int = 0
    # Older payloads predate these and load with the passing defaults.
    figures_ok: bool = True
    support: Support | Literal["unjudged"] = "supported"


class VerificationSummary(BaseModel):
    total_claims: int = 0
    kept: int = 0
    rewritten: int = 0
    dropped: int = 0
    mean_entailment_score: float = 0.0
    figure_failures: int = 0
    support_rejections: int = 0
    unjudged: int = 0
    uncited_attached: int = 0
    uncited_dropped: int = 0
    sections_removed: int = 0
    verdicts: list[ClaimVerdict] = Field(default_factory=list)


class _RewritePayload(BaseModel):
    sentence: str


class SupportVerdict(BaseModel):
    """What the triage model returns for one (sentence, evidence) pair."""

    verdict: Support


REWRITE_PROMPT = """The sentence below claims more than its cited evidence supports.

Evidence:
---
{evidence}
---

Sentence: {sentence}

Rewrite the sentence so every claim is directly supported by the evidence
above, using the evidence's own names and figures. Keep the citation markers
{markers} exactly as they are; add nothing the evidence does not say. Reply
with JSON only: {{"sentence": "..."}}"""


CHECK_PROMPT = """Does the evidence support the sentence?

Evidence:
---
{evidence}
---

Sentence: {sentence}

Answer "supported" only if the evidence states everything the sentence claims,
including every name, date, and figure, about the same subject. Answer
"partial" if it supports only part of the sentence, or the same claim about a
different subject, place, or product. Answer "unsupported" otherwise.
Reply with JSON only: {{"verdict": "supported" | "partial" | "unsupported"}}"""


@dataclass
class _Unit:
    """One verifiable sentence. ``numbers`` is empty for an uncited factual
    sentence until a marker is attached."""

    text: str
    numbers: list[int]
    evidence: str
    section_index: int
    cited: bool = True
    score: float = 0.0
    figures_ok: bool = True
    support: Support | Literal["unjudged"] = "unjudged"
    action: Action = "kept"
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


def evidence_budget_chars(cfg: QuarryConfig) -> int:
    """Characters of cited evidence one check prompt can carry safely."""
    free = cfg.triage.context_tokens - _REWRITE_MAX_TOKENS - _PROMPT_OVERHEAD_TOKENS
    return max(free, 1000) * _CHARS_PER_TOKEN


def _evidence_for(
    numbers: list[int],
    citations: CitationIndex,
    chunk_lookup: dict[str, Chunk],
    budget_chars: int,
) -> str:
    """Whole cited chunk texts; each is cut to a fair share only when together
    they exceed the budget. Empty when nothing resolves.

    A marker that does not resolve is skipped defensively here and stripped
    by cleanup and render.
    """
    texts: list[str] = []
    for number in dict.fromkeys(numbers):
        try:
            citation = citations.get(number)
        except CitationError:
            continue
        chunk = chunk_lookup.get(citation.chunk_id)
        if chunk is not None:
            texts.append(chunk.text)
    if sum(len(text) for text in texts) > budget_chars and texts:
        share = budget_chars // len(texts)
        texts = [text[:share] for text in texts]
    return "\n\n".join(texts)


def figures_supported(sentence: str, evidence: str) -> bool:
    """Every figure in the sentence appears in the evidence, normalized."""
    return extract_figures(strip_markers(sentence)) <= extract_figures(evidence)


def _attach_marker(sentence: str, number: int) -> str:
    match = _TRAILING_PUNCT.match(sentence.rstrip())
    assert match is not None  # the pattern matches any string
    body, punct = match.group(1).rstrip(), match.group(2) or ""
    return f"{body} [{number}]{punct}"


def _parse_block(
    block: str,
    section_index: int,
    attach: bool,
    citations: CitationIndex,
    chunk_lookup: dict[str, Chunk],
    budget_chars: int,
) -> _Block:
    stripped = block.strip()
    if not stripped or stripped.startswith(("#", ">", "|", "```")):
        return _Block(passthrough=stripped)
    lines = [line for line in stripped.splitlines() if line.strip()]
    is_list = all(_LIST_LINE.match(line) for line in lines)
    pieces = lines if is_list else split_sentences(" ".join(lines))
    sentences: list[_Sentence] = []
    for piece in pieces:
        numbers = [int(m.group(1)) for m in CITATION_MARKER.finditer(piece)]
        unit: _Unit | None = None
        if numbers:
            evidence = _evidence_for(numbers, citations, chunk_lookup, budget_chars)
            if evidence:
                unit = _Unit(
                    text=piece, numbers=numbers, evidence=evidence, section_index=section_index
                )
        elif attach and is_factual(piece):
            unit = _Unit(
                text=piece, numbers=[], evidence="", section_index=section_index, cited=False
            )
        sentences.append(_Sentence(text=piece, unit=unit))
    return _Block(is_list=is_list, sentences=sentences)


async def _check_support(
    unit_text: str, evidence: str, llm: LocalLLM, cfg: QuarryConfig
) -> Support | Literal["unjudged"]:
    """The model's verdict, or "unjudged" when it cannot answer after retries."""
    try:
        payload = await llm.complete_typed(
            CHECK_PROMPT.format(evidence=evidence, sentence=strip_markers(unit_text)),
            SupportVerdict,
            max_tokens=_CHECK_MAX_TOKENS,
            max_retries=cfg.triage.max_retries,
        )
    except ValueError:
        return "unjudged"
    return payload.verdict


async def _rewrite_or_drop(
    unit: _Unit, reranker: Reranker, llm: LocalLLM, cfg: QuarryConfig
) -> None:
    """Up to verify.max_rewrites attempts on the triage model. A rewrite must
    keep a non-empty subset of the original markers, clear the floor, carry
    only figures the evidence has, and be judged supported; else the sentence
    is dropped."""
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
        if not figures_supported(candidate, unit.evidence):
            continue
        (score,) = await reranker.score_pairs([(candidate, unit.evidence)])
        if score < cfg.verify.floor:
            continue
        if await _check_support(candidate, unit.evidence, llm, cfg) != "supported":
            continue
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
    """Batched by model: every reranker score first, then every model check,
    then rewrites for the failures. Returns the cleaned draft plus the audit."""
    budget = evidence_budget_chars(cfg)
    parsed: list[tuple[ReportSection, list[_Block]]] = []
    units: list[tuple[str, _Unit]] = []
    for index, section in enumerate(draft.sections):
        attach = section.title not in _NO_ATTACH_SECTIONS
        blocks = [
            _parse_block(block, index, attach, citations, chunk_lookup, budget)
            for block in re.split(r"\n\s*\n", section.markdown)
        ]
        parsed.append((section, blocks))
        for block in blocks:
            for sentence in block.sentences:
                if sentence.unit is not None:
                    units.append((section.title, sentence.unit))

    if not units:
        return draft, VerificationSummary()

    cited = [unit for _, unit in units if unit.cited]
    uncited = [unit for _, unit in units if not unit.cited]

    # Uncited sentences may only borrow a chunk their own section cites (or,
    # in a section citing nothing, one the report cites).
    report_numbers = sorted({n for unit in cited for n in unit.numbers})
    section_numbers: dict[int, list[int]] = {}
    for unit in cited:
        section_numbers.setdefault(unit.section_index, []).extend(unit.numbers)
    candidate_pairs: list[tuple[_Unit, int, str]] = []
    for unit in uncited:
        pool = sorted(set(section_numbers.get(unit.section_index, []))) or report_numbers
        for number in pool:
            evidence = _evidence_for([number], citations, chunk_lookup, budget)
            if evidence:
                candidate_pairs.append((unit, number, evidence))

    # ---- Pass 1, reranker: every cited pair and every attachment candidate.
    pairs = [(unit.text, unit.evidence) for unit in cited]
    pairs += [(strip_markers(unit.text), evidence) for unit, _, evidence in candidate_pairs]
    scores = await reranker.score_pairs(pairs) if pairs else []
    for unit, score in zip(cited, scores[: len(cited)], strict=True):
        unit.score = score
        unit.figures_ok = figures_supported(unit.text, unit.evidence)
    best: dict[int, tuple[float, int, str]] = {}
    for (unit, number, evidence), score in zip(candidate_pairs, scores[len(cited) :], strict=True):
        key = id(unit)
        if score >= cfg.verify.floor and (key not in best or score > best[key][0]):
            best[key] = (score, number, evidence)

    # ---- Pass 2, triage model: support checks.
    if (cited or best) and llama_server is not None:
        # The reranker and the triage model fit together under the budget,
        # so this is the stage's only potential llama-server swap.
        await llama_server.start()
    n_unjudged = 0
    for unit in cited:
        if unit.score >= cfg.verify.floor and unit.figures_ok:
            unit.support = await _check_support(unit.text, unit.evidence, llm, cfg)
            n_unjudged += unit.support == "unjudged"
    for unit in uncited:
        choice = best.get(id(unit))
        if choice is None:
            unit.action = "dropped"
            continue
        score, number, evidence = choice
        unit.support = await _check_support(unit.text, evidence, llm, cfg)
        if unit.support == "supported" and figures_supported(unit.text, evidence):
            unit.text = _attach_marker(unit.text, number)
            unit.numbers = [number]
            unit.evidence = evidence
            unit.score = score
        else:
            unit.action = "dropped"

    # ---- Pass 3: rewrite-or-drop every cited sentence that failed a gate.
    for unit in cited:
        if unit.score < cfg.verify.floor or not unit.figures_ok or unit.support != "supported":
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
            if kept:
                out_blocks.append(("\n" if block.is_list else " ").join(kept))
        markdown = "\n\n".join(out_blocks).strip()
        # A section emptied by drops is removed; headings alone are not content.
        if any(not line.lstrip().startswith("#") for line in markdown.splitlines() if line.strip()):
            new_sections.append(ReportSection(title=section.title, markdown=markdown))

    survivor_scores = [unit.score for unit in cited if unit.action != "dropped"]
    summary = VerificationSummary(
        total_claims=len(cited),
        kept=sum(1 for unit in cited if unit.action == "kept"),
        rewritten=sum(1 for unit in cited if unit.action == "rewritten"),
        dropped=sum(1 for unit in cited if unit.action == "dropped"),
        mean_entailment_score=(
            sum(survivor_scores) / len(survivor_scores) if survivor_scores else 0.0
        ),
        figure_failures=sum(1 for unit in cited if not unit.figures_ok),
        support_rejections=sum(1 for unit in cited if unit.support in ("partial", "unsupported")),
        unjudged=n_unjudged,
        uncited_attached=sum(1 for unit in uncited if unit.action != "dropped"),
        uncited_dropped=sum(1 for unit in uncited if unit.action == "dropped"),
        sections_removed=len(draft.sections) - len(new_sections),
        verdicts=[
            ClaimVerdict(
                section_title=title,
                sentence=unit.text,
                citation_numbers=unit.numbers,
                entailment_score=unit.score,
                action=unit.action,
                rewrite_attempts=unit.rewrite_attempts,
                figures_ok=unit.figures_ok,
                support=unit.support,
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
        figure_failures=summary.figure_failures,
        support_rejections=summary.support_rejections,
        unjudged=summary.unjudged,
        uncited_attached=summary.uncited_attached,
        uncited_dropped=summary.uncited_dropped,
        sections_removed=summary.sections_removed,
    )
    verified = DraftReport(topic=draft.topic, sections=new_sections, corpus_hash=draft.corpus_hash)
    return verified, summary
