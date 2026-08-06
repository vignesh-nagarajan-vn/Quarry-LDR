"""SYNTHESIZE stage: the report is written section by section.

Two paths share one section plan, one evidence budget, and one citation
index. ``synthesize()`` is the API path: the corpus is built once,
deterministically, hashed once, and every section call sends the byte
identical prefix with a 1h cache_control breakpoint; the provider raises
CachePrefixError on drift rather than silently paying full price.
``synthesize_local()`` is the local path: a 16K-context model cannot hold
the full corpus, so each body section sees only its own sub-questions'
evidence and Overview/Conclusions see a compact digest, while citation
numbers come from one global index assigned up front. Both paths first trim
evidence to ``report.corpus_budget_tokens`` (round-robin across
sub-questions in descending rerank order).
"""

from __future__ import annotations

import re
from collections import Counter, deque

from pydantic import BaseModel, Field

from quarry_ldr.config import QuarryConfig
from quarry_ldr.ingest.chunk import HeuristicTokenCounter, TokenCounter
from quarry_ldr.ledger import CostCapExceeded
from quarry_ldr.logging import get_logger
from quarry_ldr.pipeline.plan import ResearchPlan
from quarry_ldr.pipeline.triage import TriagedChunk
from quarry_ldr.providers.base import Provider, hash_corpus
from quarry_ldr.report.citations import CITATION_MARKER, CitationError, CitationIndex

logger = get_logger(component="synthesize")


class SectionBrief(BaseModel):
    """One planned section of the report."""

    title: str
    instructions: str
    sub_question_ids: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    title: str
    markdown: str  # cites evidence as [n] markers resolved by CitationIndex


class DraftReport(BaseModel):
    topic: str
    sections: list[ReportSection] = Field(default_factory=list)
    corpus_hash: str = ""


def _corpus_entry(item: TriagedChunk, number: int) -> str:
    """One evidence entry exactly as it ships in the corpus; also the text
    the budget counter prices, so selection and payload agree."""
    heading = " > ".join(item.chunk.heading_path) if item.chunk.heading_path else "(top)"
    return (
        f"[{number}] source: {item.chunk.url} | section: {heading}\n"
        f"claim: {item.verdict.claim}\n"
        f"evidence: {item.verdict.evidence_span}\n"
        f"text: {item.chunk.text}"
    )


def select_evidence(
    evidence: list[TriagedChunk],
    budget_tokens: int,
    counter: TokenCounter | None = None,
) -> list[TriagedChunk]:
    """Trim evidence to the synthesis token budget.

    Round-robin across sub-questions in descending rerank order, so coverage
    degrades evenly: every sub-question keeps its strongest chunks and no
    sub-question is dropped wholesale. An entry that no longer fits is
    skipped and selection continues, so one oversized chunk cannot end the
    pass early. Deterministic for a given evidence set.
    """
    active_counter: TokenCounter = counter if counter is not None else HeuristicTokenCounter()
    ordered = sorted(evidence, key=lambda t: (t.sub_question_id, -t.rerank_score, t.chunk.chunk_id))
    grouped: dict[str, deque[TriagedChunk]] = {}
    for item in ordered:
        grouped.setdefault(item.sub_question_id, deque()).append(item)
    queues = deque(grouped[sq_id] for sq_id in sorted(grouped))
    kept: list[TriagedChunk] = []
    spent = 0
    while queues:
        queue = queues.popleft()
        item = queue.popleft()
        cost = active_counter.count(_corpus_entry(item, 0))
        if spent + cost <= budget_tokens:
            kept.append(item)
            spent += cost
        if queue:
            queues.append(queue)
    return kept


def build_evidence_corpus(evidence: list[TriagedChunk], citations: CitationIndex) -> str:
    """Deterministic corpus: evidence grouped by sub-question, each chunk
    tagged with its citation number. Byte-stable across section calls."""
    ordered = sorted(evidence, key=lambda t: (t.sub_question_id, -t.rerank_score, t.chunk.chunk_id))
    lines: list[str] = ["EVIDENCE CORPUS", ""]
    current_sq = None
    for item in ordered:
        number = citations.add(item.chunk)
        if item.sub_question_id != current_sq:
            current_sq = item.sub_question_id
            lines.append(f"### Sub-question {current_sq}")
            lines.append("")
        lines.append(_corpus_entry(item, number))
        lines.append("")
    return "\n".join(lines)


def plan_sections(plan: ResearchPlan, cfg: QuarryConfig) -> list[SectionBrief]:
    """Derive section briefs from the research plan (min/max from config).

    Shape: an overview, one body section per sub-question (grouped when the
    plan has more sub-questions than the section budget), and a conclusion.
    """
    max_body = max(cfg.report.max_sections - 2, 1)
    sub_questions = plan.sub_questions
    groups: list[list[str]] = []
    if len(sub_questions) <= max_body:
        groups = [[sq.id] for sq in sub_questions]
    else:
        # Distribute sub-questions across max_body groups, round-robin-free:
        # consecutive ids stay together so sections read coherently.
        per_group = -(-len(sub_questions) // max_body)  # ceil division
        for start in range(0, len(sub_questions), per_group):
            groups.append([sq.id for sq in sub_questions[start : start + per_group]])

    by_id = {sq.id: sq for sq in sub_questions}
    briefs = [
        SectionBrief(
            title="Overview",
            instructions=(
                "Write a compact overview of the topic and the report's key findings. "
                "Cite the strongest evidence."
            ),
            sub_question_ids=[sq.id for sq in sub_questions],
        )
    ]
    for group in groups:
        questions = "; ".join(by_id[sq_id].question for sq_id in group)
        title = by_id[group[0]].question.rstrip("?")
        if len(title) > 70:
            # Cut at a word boundary; a mid-word chop with a "..." marker
            # reads broken in headings and the PDF contents page.
            title = title[:70].rsplit(" ", 1)[0]
        briefs.append(
            SectionBrief(
                title=title,
                instructions=f"Answer in depth: {questions}",
                sub_question_ids=list(group),
            )
        )
    briefs.append(
        SectionBrief(
            title="Conclusions",
            instructions=(
                "Synthesize the overall answer, note open questions and conflicting "
                "evidence explicitly."
            ),
            sub_question_ids=[sq.id for sq in sub_questions],
        )
    )
    return briefs


SECTION_PROMPT = """Write the report section titled "{title}".

{instructions}

Rules:
- Use ONLY the evidence corpus above. Every factual claim carries a citation
  marker [n] matching a corpus entry. No uncited claims.
- Markdown body text only: no top-level heading (it is added by the renderer),
  subsections with ### are fine.
- If evidence conflicts, say so and cite both sides.
- If the corpus lacks evidence for something, say the evidence is missing
  rather than inventing it.
- Relevant sub-questions: {sub_question_ids}.
"""


async def synthesize(
    plan: ResearchPlan,
    evidence: list[TriagedChunk],
    provider: Provider,
    cfg: QuarryConfig,
    citations: CitationIndex,
) -> DraftReport:
    """Section-by-section models.synthesize calls over the cached corpus."""
    selected = select_evidence(evidence, cfg.report.corpus_budget_tokens)
    if len(selected) < len(evidence):
        logger.info(
            "evidence_budget_applied",
            n_in=len(evidence),
            n_kept=len(selected),
            budget_tokens=cfg.report.corpus_budget_tokens,
        )
    corpus = build_evidence_corpus(selected, citations)
    corpus_hash = hash_corpus(corpus)
    sections: list[ReportSection] = []
    for brief in plan_sections(plan, cfg):

        async def call_section(brief: SectionBrief = brief) -> tuple[str, str | None]:
            result = await provider.complete_with_cached_corpus(
                model=cfg.models.synthesize,
                cache_name=f"synthesis:{corpus_hash[:12]}",
                corpus=corpus,
                brief=SECTION_PROMPT.format(
                    title=brief.title,
                    instructions=brief.instructions,
                    sub_question_ids=", ".join(brief.sub_question_ids),
                ),
                # claude-opus-5 thinks inside this budget; 3000 truncated every
                # section of the first passing smoke run and left one empty.
                max_tokens=8192,
                stage="synthesize",
            )
            return result.text.strip(), result.stop_reason

        markdown, stop_reason = await call_section()
        if stop_reason == "max_tokens":
            logger.warning("section_hit_max_tokens", title=brief.title)
        if not markdown:
            # The model can spend the whole budget thinking; one retry reads
            # the cached corpus, so the second attempt costs cents.
            logger.warning("section_empty_retrying", title=brief.title)
            markdown, _ = await call_section()
        sections.append(ReportSection(title=brief.title, markdown=_strip_leading_heading(markdown)))
    return DraftReport(topic=plan.topic, sections=sections, corpus_hash=corpus_hash)


class _SectionPayload(BaseModel):
    """The JSON shape the local model returns for one section."""

    markdown: str


LOCAL_SECTION_RULES = (
    '\nReply with JSON only: {"markdown": "the section body"}. Cite only [n] '
    "numbers that appear in the corpus above; never invent numbers."
)

_DIGEST_CLAIMS_PER_SQ = 3


def _digest_corpus(
    plan: ResearchPlan,
    evidence_by_sq: dict[str, list[TriagedChunk]],
    citations: CitationIndex,
) -> str:
    """Compact all-sub-question digest for the Overview/Conclusions sections:
    top claims only, tagged with their already-assigned citation numbers."""
    lines = ["EVIDENCE DIGEST (key claims per sub-question)", ""]
    for sub_question in plan.sub_questions:
        lines.append(f"### {sub_question.id}: {sub_question.question}")
        items = evidence_by_sq.get(sub_question.id, [])[:_DIGEST_CLAIMS_PER_SQ]
        if not items:
            lines.append("(no evidence collected)")
        for item in items:
            number = citations.add(item.chunk)
            lines.append(f"[{number}] {item.verdict.claim}")
        lines.append("")
    return "\n".join(lines)


def _section_corpus(
    evidence_by_sq: dict[str, list[TriagedChunk]],
    sub_question_ids: list[str],
    citations: CitationIndex,
    budget_tokens: int,
) -> str:
    """Corpus for one body section: only its sub-questions' evidence, trimmed
    with the same round-robin policy, numbered from the global index
    (``add()`` returns the existing number for chunks numbered up front)."""
    scoped = [item for sq_id in sub_question_ids for item in evidence_by_sq.get(sq_id, [])]
    trimmed = select_evidence(scoped, budget_tokens)
    return build_evidence_corpus(trimmed, citations)


_LEADING_HEADING = re.compile(r"^#{1,2}\s+[^\n]*\n?")


def _strip_leading_heading(markdown: str) -> str:
    """Drop a model-emitted top-level heading opening a section body.

    The renderer owns section headings and SECTION_PROMPT forbids them, but
    claude-opus-5 was observed opening sections with a literal `# Title` or
    `## Title` line anyway, which rendered as duplicate text under the styled
    heading. Deeper headings (###+) are legitimate subsections and pass.
    """
    return _LEADING_HEADING.sub("", markdown.lstrip(), count=1).lstrip("\n")


def _strip_invalid_citations(markdown: str, citations: CitationIndex) -> str:
    """Remove [n] markers the index does not know. Invented numbers would
    fail validate_markdown at render; semantically-wrong-but-resolving
    citations are the VERIFY stage's job, not synthesis's."""

    def _keep_or_drop(match: re.Match[str]) -> str:
        try:
            citations.get(int(match.group(1)))
        except CitationError:
            return ""
        return match.group(0)

    return CITATION_MARKER.sub(_keep_or_drop, markdown)


async def synthesize_local(
    plan: ResearchPlan,
    evidence: list[TriagedChunk],
    provider: Provider,
    cfg: QuarryConfig,
    citations: CitationIndex,
) -> DraftReport:
    """Per-section synthesis for the local engine (and the assisted draft).

    The full evidence selection is numbered into ``citations`` before any
    section is written, so numbering is stable no matter which slice each
    section sees and the references list stays complete. Section calls use
    the grammar-constrained JSON path the triage stage already proves
    against llama-server.
    """
    selected = select_evidence(evidence, cfg.report.corpus_budget_tokens)
    if len(selected) < len(evidence):
        logger.info(
            "evidence_budget_applied",
            n_in=len(evidence),
            n_kept=len(selected),
            budget_tokens=cfg.report.corpus_budget_tokens,
        )
    # Assign the global citation numbering once, up front; the returned
    # corpus string is discarded on purpose (no section ever sees all of it).
    build_evidence_corpus(selected, citations)

    ordered = sorted(selected, key=lambda t: (t.sub_question_id, -t.rerank_score, t.chunk.chunk_id))
    evidence_by_sq: dict[str, list[TriagedChunk]] = {}
    for item in ordered:
        evidence_by_sq.setdefault(item.sub_question_id, []).append(item)

    all_ids = {sq.id for sq in plan.sub_questions}
    digest = _digest_corpus(plan, evidence_by_sq, citations)
    sections: list[ReportSection] = []
    for brief in plan_sections(plan, cfg):
        # Overview/Conclusions span every sub-question (detected structurally,
        # not by title): they get the digest, never a full corpus.
        if set(brief.sub_question_ids) == all_ids:
            corpus = digest
        else:
            corpus = _section_corpus(
                evidence_by_sq, brief.sub_question_ids, citations, cfg.synth.section_budget_tokens
            )
        prompt = (
            corpus
            + "\n\n"
            + SECTION_PROMPT.format(
                title=brief.title,
                instructions=brief.instructions,
                sub_question_ids=", ".join(brief.sub_question_ids),
            )
            + LOCAL_SECTION_RULES
        )

        async def call_section(section_prompt: str = prompt) -> str:
            payload = await provider.complete_typed(
                model=cfg.models.synthesize,
                prompt=section_prompt,
                schema=_SectionPayload,
                max_tokens=cfg.synth.section_max_tokens,
                stage="synthesize",
            )
            return payload.markdown.strip()

        markdown = await call_section()
        if not markdown:
            logger.warning("section_empty_retrying", title=brief.title)
            markdown = await call_section()
        cleaned = _strip_invalid_citations(_strip_leading_heading(markdown), citations)
        n_stripped = len(citations.numbers_in(markdown)) - len(citations.numbers_in(cleaned))
        if n_stripped:
            logger.warning("section_invalid_citations_stripped", title=brief.title, n=n_stripped)
        sections.append(ReportSection(title=brief.title, markdown=cleaned))
    # No single shared prefix exists on the local path; an empty hash keeps
    # anything downstream from mistaking it for a cache key.
    return DraftReport(topic=plan.topic, sections=sections, corpus_hash="")


_POLISH_DELIMITER = re.compile(r"<!--Q-SECTION:(.*?)-->")

POLISH_SYSTEM = (
    "You polish research report drafts. Improve prose quality: smooth "
    "transitions, tighten wording, fix grammar and repetition. Hard rules:\n"
    "- Never add, remove, renumber, or relocate citation markers like [3]; "
    "every marker stays attached to the claim it supports.\n"
    "- Never add or remove facts.\n"
    "- Keep every <!--Q-SECTION:...--> delimiter line exactly as it is, in "
    "place, starting the reply with the first delimiter.\n"
    "- Keep markdown structure (### subsections, lists).\n"
    "Return the full polished document and nothing else."
)


def _marker_multiset(text: str) -> Counter[int]:
    return Counter(int(match.group(1)) for match in CITATION_MARKER.finditer(text))


async def polish_draft(draft: DraftReport, provider: Provider, cfg: QuarryConfig) -> DraftReport:
    """One models.assisted call over the assembled draft (assisted mode only).

    The citation-marker multiset must survive exactly and the section
    delimiters must round-trip in order with nothing before the first one;
    any violation, or any API failure short of the cost cap, discards the
    polish with a warning and the local draft stands. Polish is an
    enhancement, never a run-killer.
    """
    delimited: list[str] = []
    for section in draft.sections:
        delimited.append(f"<!--Q-SECTION:{section.title}-->")
        delimited.append(section.markdown)
    assembled = "\n\n".join(delimited)
    before = _marker_multiset(assembled)
    try:
        result = await provider.complete(
            model=cfg.models.assisted,
            system=POLISH_SYSTEM,
            prompt=assembled,
            max_tokens=8192,
            stage="polish",
        )
    except CostCapExceeded:
        raise  # budget enforcement always propagates
    except Exception as exc:
        logger.warning("polish_failed", error=str(exc)[:200])
        return draft

    text = result.text.strip()
    if _marker_multiset(text) != before:
        logger.warning("polish_discarded", reason="citation markers changed")
        return draft
    parts = _POLISH_DELIMITER.split(text)
    prefix, titles, bodies = parts[0], parts[1::2], parts[2::2]
    if prefix.strip() or titles != [section.title for section in draft.sections]:
        logger.warning("polish_discarded", reason="section delimiters did not round-trip")
        return draft
    logger.info("polish_applied", n_sections=len(titles))
    return DraftReport(
        topic=draft.topic,
        sections=[
            ReportSection(title=title, markdown=body.strip())
            for title, body in zip(titles, bodies, strict=True)
        ],
        corpus_hash=draft.corpus_hash,
    )
