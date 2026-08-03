"""TRIAGE stage: the local 4B model filters reranked chunks into evidence.

High volume, low intelligence: per-chunk structured JSON verdicts
{relevant, claim, evidence_span, confidence}, validated by pydantic with a
retry-on-malformed path in LocalLLM.complete_typed.

Implemented in M6.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from quarry_ldr.config import TriageSettings
from quarry_ldr.gpu.local_llm import LocalLLM
from quarry_ldr.gpu.reranker import ScoredChunk
from quarry_ldr.ingest.chunk import Chunk
from quarry_ldr.logging import get_logger
from quarry_ldr.pipeline.plan import SubQuestion


class TriageVerdict(BaseModel):
    relevant: bool
    claim: str = ""
    evidence_span: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class TriagedChunk(BaseModel):
    """A chunk that survived triage for one sub-question."""

    sub_question_id: str
    chunk: Chunk
    verdict: TriageVerdict
    rerank_score: float


TRIAGE_PROMPT = """You are filtering evidence for a research question.

Question: {question}

Passage:
---
{passage}
---

Reply with JSON only, exactly this shape:
{{"relevant": true/false, "claim": "one-sentence factual claim the passage supports, empty if irrelevant", "evidence_span": "shortest verbatim quote supporting the claim, empty if irrelevant", "confidence": 0.0-1.0}}
"""


async def triage_chunks(
    sub_question: SubQuestion,
    scored: list[ScoredChunk],
    llm: LocalLLM,
    settings: TriageSettings,
) -> list[TriagedChunk]:
    """Run every chunk through the local model; keep relevant verdicts at or
    above settings.confidence_floor. Malformed-JSON chunks are retried, then
    dropped (never crash the run over one bad chunk)."""
    logger = get_logger(component="triage", sub_question=sub_question.id)
    kept: list[TriagedChunk] = []
    n_dropped_malformed = 0
    n_dropped_verdict = 0
    for item in scored:
        prompt = TRIAGE_PROMPT.format(
            question=sub_question.question, passage=item.chunk.text[:6000]
        )
        try:
            verdict = await llm.complete_typed(
                prompt, TriageVerdict, max_retries=settings.max_retries
            )
        except ValueError:
            n_dropped_malformed += 1
            logger.warning("triage_chunk_dropped_malformed", chunk_id=item.chunk.chunk_id)
            continue
        if verdict.relevant and verdict.confidence >= settings.confidence_floor:
            kept.append(
                TriagedChunk(
                    sub_question_id=sub_question.id,
                    chunk=item.chunk,
                    verdict=verdict,
                    rerank_score=item.score,
                )
            )
        else:
            n_dropped_verdict += 1
    logger.info(
        "triage_done",
        n_in=len(scored),
        n_kept=len(kept),
        n_dropped_verdict=n_dropped_verdict,
        n_dropped_malformed=n_dropped_malformed,
    )
    return kept
