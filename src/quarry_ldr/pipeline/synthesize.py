"""SYNTHESIZE stage: the report is written section by section.

Two paths share one section plan, one evidence budget, and one citation
index. ``synthesize()`` is the API path: the corpus is built once,
deterministically, hashed once, and every section call sends the byte
identical prefix with a 1h cache_control breakpoint; the provider raises
CachePrefixError on drift rather than silently paying full price.
``synthesize_local()`` is the local path: a 16K-context model cannot hold
the full corpus, so each body section sees only its own sub-questions'
evidence, while citation numbers come from one global index assigned up front. Both paths first trim
evidence to ``report.corpus_budget_tokens`` (round-robin across
sub-questions in descending rerank order).
"""

from __future__ import annotations

import re
from collections import Counter, deque

from pydantic import BaseModel, Field

from quarry_ldr.config import QuarryConfig, SourceSettings
from quarry_ldr.ingest.chunk import HeuristicTokenCounter, TokenCounter
from quarry_ldr.ingest.sources import adjusted_score, site_of
from quarry_ldr.ledger import CostCapExceeded
from quarry_ldr.logging import get_logger
from quarry_ldr.pipeline.conflicts import Conflict, conflict_notes
from quarry_ldr.pipeline.plan import ResearchPlan, SectionPlan
from quarry_ldr.pipeline.triage import TriagedChunk
from quarry_ldr.providers.base import Provider, hash_corpus
from quarry_ldr.report.citations import CITATION_MARKER, CitationError, CitationIndex
from quarry_ldr.report.claims import count_specifics

logger = get_logger(component="synthesize")

# Enough to settle near-ties toward concrete claims, never enough to outrank
# a real relevance difference (reranker logits span about 24 points).
_SPECIFICS_BONUS = 0.05
_SPECIFICS_BONUS_CAP = 6


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
    source: SourceSettings | None = None,
) -> list[TriagedChunk]:
    """Trim evidence to the synthesis token budget.

    Round-robin across sub-questions in descending rerank order, so coverage
    degrades evenly: every sub-question keeps its strongest chunks and no
    sub-question is dropped wholesale. An entry that no longer fits is
    skipped and selection continues, so one oversized chunk cannot end the
    pass early. Deterministic for a given evidence set.

    With ``source``, order follows the tier-adjusted score and no site spends
    more than its share of the budget. The share widens to ``1 / n_sites``
    when too few sites exist for the configured cap to be possible, so a thin
    corpus keeps its evidence.
    """
    active_counter: TokenCounter = counter if counter is not None else HeuristicTokenCounter()

    def rank(item: TriagedChunk) -> float:
        if source is None:
            return item.rerank_score
        concrete = min(count_specifics(item.verdict.claim), _SPECIFICS_BONUS_CAP)
        return adjusted_score(item.chunk.url, item.rerank_score, source) + (
            concrete * _SPECIFICS_BONUS
        )

    ordered = sorted(evidence, key=lambda t: (t.sub_question_id, -rank(t), t.chunk.chunk_id))
    grouped: dict[str, deque[TriagedChunk]] = {}
    for item in ordered:
        grouped.setdefault(item.sub_question_id, deque()).append(item)
    queues = deque(grouped[sq_id] for sq_id in sorted(grouped))

    site_budget = float(budget_tokens)
    if source is not None and evidence:
        n_sites = len({site_of(item.chunk.url) for item in evidence})
        site_budget = budget_tokens * max(source.max_site_share, 1 / n_sites)
    site_spent: dict[str, int] = {}

    kept: list[TriagedChunk] = []
    spent = 0
    while queues:
        queue = queues.popleft()
        item = queue.popleft()
        cost = active_counter.count(_corpus_entry(item, 0))
        site = site_of(item.chunk.url)
        within_site = site_spent.get(site, 0) + cost <= site_budget
        if spent + cost <= budget_tokens and within_site:
            kept.append(item)
            spent += cost
            site_spent[site] = site_spent.get(site, 0) + cost
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


_MAX_TITLE_CHARS = 60
_MAX_TITLE_WORDS = 8
_DANGLING = frozenset({"a", "an", "and", "at", "by", "for", "in", "of", "on", "or", "the", "to"})
_QUESTION_OPENING = re.compile(
    r"^(?:what|which|how|why|where|when|who)\s+"
    r"(?:(?:are|is|do|does|did|can|could|will|would|should|has|have|was|were)\s+)?(?:the\s+)?",
    re.IGNORECASE,
)
# Drafts aim above the configured minimum: VERIFY removes sentences afterward,
# and the first live run lost enough to land at 2,780 words against a 3,000 floor.
_LENGTH_MARGIN = 1.25


def _fallback_title(question: str) -> str:
    """A question reworded as a noun phrase. When it must be cut, the cut falls
    before its last trailing phrase, so it never stops mid-phrase."""
    words = _QUESTION_OPENING.sub("", question.rstrip("?")).split() or question.rstrip("?").split()
    cut = False
    while len(" ".join(words)) > _MAX_TITLE_CHARS:
        words.pop()
        cut = True
    if cut:
        for index in range(len(words) - 1, 0, -1):
            if words[index].lower() in _DANGLING:
                words = words[:index]
                break
    while len(words) > 1 and words[-1].lower() in _DANGLING:
        words.pop()
    title = " ".join(words)
    return title[:1].upper() + title[1:]


def _valid_title(title: str) -> bool:
    title = title.strip()
    return (
        0 < len(title) <= _MAX_TITLE_CHARS
        and len(title.split()) <= _MAX_TITLE_WORDS
        and not title.endswith("?")
    )


def _assign_once(plan: ResearchPlan) -> list[SectionPlan]:
    """The proposed sections with every sub-question in exactly one of them.

    The last section to list an id keeps it, unknown ids go, a section left
    empty goes, and a sub-question nobody listed joins the section holding the
    one before it. The live 8B grouped well but ran each section's last ids
    into the next one, which named them better, and missed one id; rejecting
    that threw away every good title it had written.
    """
    ids = [sq.id for sq in plan.sub_questions]
    last_listed: dict[str, int] = {}
    for position, proposed in enumerate(plan.sections):
        last_listed.update((i, position) for i in proposed.sub_question_ids if i in ids)
    owner: dict[str, SectionPlan] = {}
    sections: list[SectionPlan] = []
    for position, proposed in enumerate(plan.sections):
        kept = [
            i for i in dict.fromkeys(proposed.sub_question_ids) if last_listed.get(i) == position
        ]
        if kept:
            section = SectionPlan(title=proposed.title.strip(), sub_question_ids=kept)
            sections.append(section)
            owner.update(dict.fromkeys(kept, section))
    if not sections:
        return []
    for index, sq_id in enumerate(ids):
        if sq_id not in owner:
            section = owner[ids[index - 1]] if index else sections[0]
            section.sub_question_ids.append(sq_id)
            owner[sq_id] = section
    return sections


def section_groups(plan: ResearchPlan, max_body: int) -> list[SectionPlan]:
    """The plan's own sections when code accepts them, else a deterministic split.

    Code assigns every sub-question to exactly one section first. The result
    is accepted only if the count fits ``max_body`` and every title is short,
    unique, and not a question. Anything else falls back to consecutive groups.
    """
    ids = [sq.id for sq in plan.sub_questions]
    proposed = _assign_once(plan)
    titles = [section.title for section in proposed]
    if (
        proposed
        and len(proposed) <= max_body
        and all(_valid_title(title) for title in titles)
        and len({title.lower() for title in titles}) == len(titles)
    ):
        return proposed
    if plan.sections:
        logger.warning("plan_sections_rejected", n_proposed=len(plan.sections))
    by_id = {sq.id: sq for sq in plan.sub_questions}
    per_group = max(1, -(-len(ids) // max_body))  # ceil division; consecutive ids stay together
    return [
        SectionPlan(
            title=_fallback_title(by_id[ids[start]].question),
            sub_question_ids=ids[start : start + per_group],
        )
        for start in range(0, len(ids), per_group)
    ]


def section_word_range(cfg: QuarryConfig, n_sections: int) -> tuple[int, int]:
    """Per-section word bounds that put the report inside the configured total."""
    n = max(n_sections, 1)
    high = cfg.report.target_words_max // n
    return min(int(cfg.report.target_words_min * _LENGTH_MARGIN) // n, high), high


def plan_sections(plan: ResearchPlan, cfg: QuarryConfig) -> list[SectionBrief]:
    """One body brief per validated plan section.

    Key findings, disagreements, gaps, and sources are added around the body
    after it is written; no section here spans the whole plan.
    """
    groups = section_groups(plan, max(cfg.report.max_sections - 2, 1))
    by_id = {sq.id: sq for sq in plan.sub_questions}
    low, high = section_word_range(cfg, len(groups))
    briefs: list[SectionBrief] = []
    for group in groups:
        questions = "; ".join(by_id[sq_id].question for sq_id in group.sub_question_ids)
        briefs.append(
            SectionBrief(
                title=group.title,
                instructions=f"Answer in depth: {questions} Write {low} to {high} words.",
                sub_question_ids=list(group.sub_question_ids),
            )
        )
    return briefs


SECTION_PROMPT = """Write the report section titled "{title}".

{instructions}

Rules:
- Use ONLY the sources reviewed above. Every factual claim carries a citation
  marker [n] matching a numbered source entry. No uncited claims.
- Markdown body text only: no top-level heading (it is added by the renderer),
  subsections with ### are fine.
- Be concrete. Name the companies, projects, places, and people the evidence
  names, and give its dates and figures. Never write "some companies" or "recent
  studies" when the evidence says which.
- If evidence conflicts, say so and cite both sides.
- If the sources reviewed lack evidence for something, say so rather than
  inventing it. Call them "the sources reviewed", never "the corpus".
- Answer the questions; do not restate them.
- Relevant sub-questions: {sub_question_ids}. Never write these ids in the text.
"""


async def synthesize(
    plan: ResearchPlan,
    evidence: list[TriagedChunk],
    provider: Provider,
    cfg: QuarryConfig,
    citations: CitationIndex,
    conflicts: list[Conflict] | None = None,
) -> DraftReport:
    """Section-by-section models.synthesize calls over the cached corpus."""
    selected = select_evidence(evidence, cfg.report.corpus_budget_tokens, source=cfg.source)
    if len(selected) < len(evidence):
        logger.info(
            "evidence_budget_applied",
            n_in=len(evidence),
            n_kept=len(selected),
            budget_tokens=cfg.report.corpus_budget_tokens,
        )
    corpus = build_evidence_corpus(selected, citations)
    notes = conflict_notes(conflicts or [], citations)
    if notes:
        corpus = f"{corpus}\n{notes}\n"
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
    "numbers that appear in the sources above; never invent numbers."
)


def _section_corpus(
    evidence_by_sq: dict[str, list[TriagedChunk]],
    sub_question_ids: list[str],
    citations: CitationIndex,
    budget_tokens: int,
    source: SourceSettings,
) -> str:
    """Corpus for one body section: only its sub-questions' evidence, trimmed
    with the same round-robin policy, numbered from the global index
    (``add()`` returns the existing number for chunks numbered up front)."""
    scoped = [item for sq_id in sub_question_ids for item in evidence_by_sq.get(sq_id, [])]
    trimmed = select_evidence(scoped, budget_tokens, source=source)
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
    conflicts: list[Conflict] | None = None,
) -> DraftReport:
    """Per-section synthesis for the local engine (and the assisted draft).

    The full evidence selection is numbered into ``citations`` before any
    section is written, so numbering is stable no matter which slice each
    section sees and the references list stays complete. Section calls use
    the grammar-constrained JSON path the triage stage already proves
    against llama-server.
    """
    selected = select_evidence(evidence, cfg.report.corpus_budget_tokens, source=cfg.source)
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

    sections: list[ReportSection] = []
    for brief in plan_sections(plan, cfg):
        corpus = _section_corpus(
            evidence_by_sq,
            brief.sub_question_ids,
            citations,
            cfg.synth.section_budget_tokens,
            cfg.source,
        )
        notes = conflict_notes(conflicts or [], citations, brief.sub_question_ids)
        prompt = (
            corpus
            + (f"\n{notes}\n" if notes else "")
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


LOCAL_EDIT_PROMPT = """Edit this report section titled "{title}" for readability.

Section:
{markdown}

Rules:
- Tighten wording, smooth transitions, fix grammar, and remove repetition.
- Keep every citation marker like [3] exactly as written, attached to the
  same claim. Never add, drop, or renumber a marker.
- Never add or remove facts, names, dates, or figures.
- Keep the markdown structure (### subsections, lists).
- Refer to "the sources reviewed", never "the corpus".
"""
# An edit that shrinks or grows the section this much is a rewrite, not an edit.
_EDIT_MIN_RATIO = 0.6
_EDIT_MAX_RATIO = 1.3


async def edit_sections_local(
    draft: DraftReport, provider: Provider, cfg: QuarryConfig
) -> DraftReport:
    """One guarded local edit per section (local engine only).

    An edit stands only if its citation-marker multiset is unchanged and its
    length stays within 0.6x to 1.3x of the draft; otherwise, or on any
    failure, that section keeps its draft text.
    """
    sections: list[ReportSection] = []
    n_applied = 0
    for section in draft.sections:
        try:
            payload = await provider.complete_typed(
                model=cfg.models.synthesize,
                prompt=LOCAL_EDIT_PROMPT.format(title=section.title, markdown=section.markdown)
                + LOCAL_SECTION_RULES,
                schema=_SectionPayload,
                max_tokens=cfg.synth.section_max_tokens,
                stage="edit",
            )
            edited = _strip_leading_heading(payload.markdown.strip())
        except CostCapExceeded:
            raise
        except Exception as exc:  # an edit is an enhancement, never a run-killer
            logger.warning("edit_failed", title=section.title, error=str(exc)[:200])
            sections.append(section)
            continue
        ratio = len(edited.split()) / max(len(section.markdown.split()), 1)
        if _marker_multiset(edited) != _marker_multiset(section.markdown):
            logger.warning("edit_discarded", title=section.title, reason="citation markers changed")
            sections.append(section)
        elif not _EDIT_MIN_RATIO <= ratio <= _EDIT_MAX_RATIO:
            logger.warning(
                "edit_discarded", title=section.title, reason="length", ratio=round(ratio, 2)
            )
            sections.append(section)
        else:
            n_applied += 1
            sections.append(section.model_copy(update={"markdown": edited}))
    logger.info("edit_done", n_sections=len(sections), n_applied=n_applied)
    return draft.model_copy(update={"sections": sections})


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
