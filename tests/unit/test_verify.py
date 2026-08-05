"""VERIFY: unit extraction, entailment gating, rewrite-or-drop, reassembly."""

from __future__ import annotations

from typing import Any, cast

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.local_llm import LlamaServer, LocalLLM
from quarry_ldr.gpu.reranker import Reranker
from quarry_ldr.ingest.chunk import Chunk, make_chunk_id
from quarry_ldr.pipeline.synthesize import DraftReport, ReportSection
from quarry_ldr.pipeline.verify import VerificationSummary, verify_report
from quarry_ldr.report.citations import CitationIndex


def make_chunk(text: str, position: int = 0) -> Chunk:
    url = "https://vesterholm-times.example/2024/article"
    return Chunk(
        chunk_id=make_chunk_id(url, position),
        url=url,
        doc_title="Test doc",
        heading_path=["Background"],
        text=text,
        token_count=len(text) // 4,
        position=position,
        start_char=0,
        end_char=len(text),
    )


def build_fixture(
    markdown: str, n_chunks: int = 2
) -> tuple[DraftReport, CitationIndex, dict[str, Chunk]]:
    citations = CitationIndex()
    chunk_lookup: dict[str, Chunk] = {}
    for i in range(n_chunks):
        chunk = make_chunk(f"The pilot stores heat at 600 degrees. Fact {i}.", position=i)
        citations.add(chunk)
        chunk_lookup[chunk.chunk_id] = chunk
    draft = DraftReport(topic="t", sections=[ReportSection(title="S", markdown=markdown)])
    return draft, citations, chunk_lookup


class FakeScorer:
    """score_pairs by sentence-prefix rule; unmatched sentences score default."""

    def __init__(self, scores: dict[str, float] | None = None, default: float = 5.0) -> None:
        self.scores = scores or {}
        self.default = default
        self.calls: list[list[tuple[str, str]]] = []

    async def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        return [
            next(
                (value for prefix, value in self.scores.items() if pair[0].startswith(prefix)),
                self.default,
            )
            for pair in pairs
        ]


class FakeRewriteLLM:
    def __init__(self, rewritten: str = "A supported claim [1].") -> None:
        self.rewritten = rewritten
        self.calls = 0

    async def complete_typed(
        self, prompt: str, schema: Any, max_tokens: int = 512, max_retries: int = 2
    ) -> Any:
        self.calls += 1
        return schema.model_validate({"sentence": self.rewritten})


class ExplodingRewriteLLM:
    async def complete_typed(
        self, prompt: str, schema: Any, max_tokens: int = 512, max_retries: int = 2
    ) -> Any:
        raise ValueError("malformed every time")


class FakeServer:
    def __init__(self) -> None:
        self.started = 0

    async def start(self) -> None:
        self.started += 1


async def test_above_floor_kept_verbatim(cfg: QuarryConfig) -> None:
    draft, citations, lookup = build_fixture("### Sub\n\nGood claim [1]. Another good [2].")
    scorer, llm, server = FakeScorer(), FakeRewriteLLM(), FakeServer()
    verified, summary = await verify_report(
        draft,
        citations,
        lookup,
        cast(Reranker, scorer),
        cast(LocalLLM, llm),
        cast(LlamaServer, server),
        cfg,
    )
    assert verified.sections[0].markdown == "### Sub\n\nGood claim [1]. Another good [2]."
    assert summary.total_claims == 2
    assert summary.kept == 2
    assert summary.rewritten == 0 and summary.dropped == 0
    assert summary.mean_entailment_score == 5.0
    assert llm.calls == 0
    assert server.started == 0  # no failures: the triage server is never touched
    assert len(scorer.calls) == 1  # one batched pass over the whole report


async def test_below_floor_rewritten(cfg: QuarryConfig) -> None:
    draft, citations, lookup = build_fixture("Bad claim [1]. Good [2].")
    scorer = FakeScorer(scores={"Bad claim": -5.0})
    llm, server = FakeRewriteLLM("A supported claim [1]."), FakeServer()
    verified, summary = await verify_report(
        draft,
        citations,
        lookup,
        cast(Reranker, scorer),
        cast(LocalLLM, llm),
        cast(LlamaServer, server),
        cfg,
    )
    assert "A supported claim [1]." in verified.sections[0].markdown
    assert "Bad claim" not in verified.sections[0].markdown
    assert "Good [2]." in verified.sections[0].markdown
    assert summary.rewritten == 1 and summary.kept == 1 and summary.dropped == 0
    assert server.started == 1  # the rewrite pass asserted triage residency
    verdict = next(v for v in summary.verdicts if v.action == "rewritten")
    assert verdict.rewrite_attempts == 1
    citations.validate_markdown(verified.sections[0].markdown)


async def test_rewrite_inventing_markers_is_rejected_then_dropped(cfg: QuarryConfig) -> None:
    draft, citations, lookup = build_fixture("Bad claim [1]. Good [2].")
    scorer = FakeScorer(scores={"Bad claim": -5.0})
    llm = FakeRewriteLLM("Fabricated [7].")  # [7] is not among the original markers
    verified, summary = await verify_report(
        draft,
        citations,
        lookup,
        cast(Reranker, scorer),
        cast(LocalLLM, llm),
        None,
        cfg,
    )
    assert summary.dropped == 1
    assert "Bad claim" not in verified.sections[0].markdown
    assert "Fabricated" not in verified.sections[0].markdown
    assert verified.sections[0].markdown == "Good [2]."
    assert llm.calls == cfg.verify.max_rewrites


async def test_rewrite_valueerror_exhausts_to_drop(cfg: QuarryConfig) -> None:
    draft, citations, lookup = build_fixture("Bad claim [1].")
    scorer = FakeScorer(scores={"Bad claim": -5.0})
    verified, summary = await verify_report(
        draft,
        citations,
        lookup,
        cast(Reranker, scorer),
        cast(LocalLLM, ExplodingRewriteLLM()),
        None,
        cfg,
    )
    assert summary.dropped == 1
    assert verified.sections[0].markdown == ""


async def test_unresolvable_markers_are_skipped_not_crashed(cfg: QuarryConfig) -> None:
    draft, citations, lookup = build_fixture("Claim with unknown marker [99].")
    scorer = FakeScorer()
    verified, summary = await verify_report(
        draft,
        citations,
        lookup,
        cast(Reranker, scorer),
        cast(LocalLLM, FakeRewriteLLM()),
        None,
        cfg,
    )
    # Nothing resolved, so nothing was verifiable; render's validate_markdown
    # remains the authority on genuinely broken markers.
    assert summary == VerificationSummary()
    assert verified.sections[0].markdown == draft.sections[0].markdown
    assert scorer.calls == []


async def test_structure_passes_through_and_lists_drop_by_line(cfg: QuarryConfig) -> None:
    markdown = (
        "### Findings\n\n"
        "> A quoted passage stays.\n\n"
        "- kept item [1]\n- doomed item [2]\n\n"
        "Plain uncited sentence stays."
    )
    draft, citations, lookup = build_fixture(markdown)
    scorer = FakeScorer(scores={"- doomed item": -5.0})
    verified, summary = await verify_report(
        draft,
        citations,
        lookup,
        cast(Reranker, scorer),
        cast(LocalLLM, ExplodingRewriteLLM()),
        None,
        cfg,
    )
    out = verified.sections[0].markdown
    assert "### Findings" in out
    assert "> A quoted passage stays." in out
    assert "- kept item [1]" in out
    assert "doomed item" not in out
    assert "Plain uncited sentence stays." in out
    assert summary.total_claims == 2
    assert summary.dropped == 1
