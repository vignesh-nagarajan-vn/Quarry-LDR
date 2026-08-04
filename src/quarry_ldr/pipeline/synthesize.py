"""SYNTHESIZE stage: Opus writes the report section by section over one
prompt-cached evidence corpus.

Evidence is first trimmed to ``report.corpus_budget_tokens`` (round-robin
across sub-questions in descending rerank order), keeping the API payload at
the design's ~60K tokens no matter how permissive triage was. The corpus is
then built once, deterministically, hashed once, and every section call sends
the byte identical prefix with a 1h cache_control breakpoint; the provider
raises CachePrefixError on any drift rather than silently paying full price.
"""

from __future__ import annotations

from collections import deque

from pydantic import BaseModel, Field

from quarry_ldr.config import QuarryConfig
from quarry_ldr.ingest.chunk import HeuristicTokenCounter, TokenCounter
from quarry_ldr.logging import get_logger
from quarry_ldr.pipeline.plan import ResearchPlan
from quarry_ldr.pipeline.triage import TriagedChunk
from quarry_ldr.providers.anthropic_client import AnthropicProvider, hash_corpus
from quarry_ldr.report.citations import CitationIndex

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
            title = title[:67] + "..."
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
    provider: AnthropicProvider,
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
        sections.append(ReportSection(title=brief.title, markdown=markdown))
    return DraftReport(topic=plan.topic, sections=sections, corpus_hash=corpus_hash)
