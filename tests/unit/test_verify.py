"""VERIFY: unit extraction, entailment gating, rewrite-or-drop, reassembly."""

from __future__ import annotations

from typing import Any, cast

import pytest

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.local_llm import LlamaServer, LocalLLM
from quarry_ldr.gpu.reranker import Reranker
from quarry_ldr.ingest.chunk import Chunk, make_chunk_id
from quarry_ldr.pipeline.findings import GAPS_TITLE
from quarry_ldr.pipeline.synthesize import DraftReport, ReportSection
from quarry_ldr.pipeline.verify import (
    SupportVerdict,
    VerificationSummary,
    _finish_rewrite,
    evidence_budget_chars,
    figures_supported,
    verify_report,
)
from quarry_ldr.report.citations import CitationIndex


@pytest.fixture(autouse=True)
def _pin_floor(cfg: QuarryConfig) -> None:
    """Pin the floor to 0.0 so the fake scores (+5 pass, -5 fail) keep their
    meaning regardless of where calibration moves the shipped default."""
    cfg.verify.floor = 0.0


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
    """Serves both VERIFY prompts: rewrites return ``rewritten``; support
    checks answer per sentence substring in ``verdicts``, else ``default``."""

    def __init__(
        self,
        rewritten: str = "A supported claim [1].",
        verdicts: dict[str, str] | None = None,
        default: str = "supported",
    ) -> None:
        self.rewritten = rewritten
        self.verdicts = verdicts or {}
        self.default = default
        self.calls = 0  # rewrite calls
        self.check_prompts: list[str] = []

    async def complete_typed(
        self, prompt: str, schema: Any, max_tokens: int = 512, max_retries: int = 2
    ) -> Any:
        if schema is SupportVerdict:
            self.check_prompts.append(prompt)
            sentence = prompt.split("Sentence: ", 1)[1].split("\n", 1)[0]
            verdict = next((v for k, v in self.verdicts.items() if k in sentence), self.default)
            if verdict == "malformed":
                raise ValueError("no valid JSON after retries")
            return SupportVerdict(verdict=verdict)  # type: ignore[arg-type]
        self.calls += 1
        return schema.model_validate({"sentence": self.rewritten})


@pytest.mark.parametrize(
    ("original", "rewritten", "expected"),
    [
        ("A claim [1]. Next one.", "A narrower claim [1]", "A narrower claim [1]."),
        ("A claim [1]. Next one.", "A narrower claim. [1]", "A narrower claim [1]."),
        ("A claim [1]. Next one.", "A narrower claim [1][2]", "A narrower claim [1][2]."),
        ("A claim [1]. Next one.", "A narrower [1] claim", "A narrower [1] claim."),
        ("A claim [1]. Next one.", "A narrower claim [1].", "A narrower claim [1]."),
        ("- A claim [1].", "A narrower claim [1].", "- A narrower claim [1]."),
        ("- A claim [1].", "- A narrower claim [1]", "- A narrower claim [1]."),
    ],
)
def test_finish_rewrite_restores_bullets_and_sentence_ends(
    original: str, rewritten: str, expected: str
) -> None:
    assert _finish_rewrite(rewritten, original) == expected


async def test_a_rewritten_finding_stays_a_bullet_and_ends_its_sentence(cfg: QuarryConfig) -> None:
    draft, citations, lookup = build_fixture("- A finding here [1].\n- Another finding [2].")
    llm = FakeRewriteLLM(rewritten="A narrower finding [1]", verdicts={"A finding here": "partial"})
    verified, summary = await verify_report(
        draft,
        citations,
        lookup,
        cast(Reranker, FakeScorer()),
        cast(LocalLLM, llm),
        None,
        cfg,
    )
    assert summary.rewritten == 1
    assert verified.sections[0].markdown == "- A narrower finding [1].\n- Another finding [2]."


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
    assert llm.calls == 0  # no rewrites
    assert len(llm.check_prompts) == 2  # every cited sentence gets a support check
    assert server.started == 1  # the checks need the triage model, once
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
    assert verified.sections == []  # a section emptied by drops is removed
    assert summary.sections_removed == 1


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
        cast(LocalLLM, FakeRewriteLLM("Fabricated [7].")),
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


async def _verify(
    markdown: str, llm: Any, cfg: QuarryConfig, scorer: FakeScorer | None = None, title: str = "S"
) -> tuple[DraftReport, VerificationSummary]:
    draft, citations, lookup = build_fixture(markdown)
    draft = draft.model_copy(update={"sections": [ReportSection(title=title, markdown=markdown)]})
    return await verify_report(
        draft,
        citations,
        lookup,
        cast(Reranker, scorer or FakeScorer()),
        cast(LocalLLM, llm),
        None,
        cfg,
    )


def test_figures_must_appear_in_the_evidence() -> None:
    evidence = "The pilot stores heat at 600 degrees and cost 1,200 euros."
    assert figures_supported("Heat reaches 600 degrees [1].", evidence)
    assert figures_supported("It cost 1200 euros [1].", evidence)
    assert not figures_supported("Heat reaches 900 degrees [1].", evidence)


async def test_a_planted_wrong_number_is_rewritten_to_the_evidence(cfg: QuarryConfig) -> None:
    llm = FakeRewriteLLM("The pilot stores heat at 600 degrees [1].")
    verified, summary = await _verify("The pilot stores heat at 900 degrees [1].", llm, cfg)
    assert verified.sections[0].markdown == "The pilot stores heat at 600 degrees [1]."
    assert summary.figure_failures == 1 and summary.rewritten == 1
    assert llm.check_prompts and "900" not in llm.check_prompts[-1].split("Sentence: ")[1]


async def test_a_wrong_number_that_survives_rewriting_is_dropped(cfg: QuarryConfig) -> None:
    llm = FakeRewriteLLM("The pilot stores heat at 950 degrees [1].")
    verified, summary = await _verify(
        "Fine claim [2]. The pilot stores heat at 900 degrees [1].", llm, cfg
    )
    assert verified.sections[0].markdown == "Fine claim [2]."
    assert summary.dropped == 1 and summary.figure_failures == 1


async def test_partial_support_goes_to_rewrite_and_unsupported_rewrites_drop(
    cfg: QuarryConfig,
) -> None:
    llm = FakeRewriteLLM(
        "The pilot is the largest in Europe [1].",
        verdicts={"Norway": "partial", "largest": "unsupported"},
    )
    verified, summary = await _verify("Fine [2]. The pilot is in Norway [1].", llm, cfg)
    assert verified.sections[0].markdown == "Fine [2]."
    assert summary.support_rejections == 1
    assert summary.dropped == 1
    assert llm.calls == cfg.verify.max_rewrites


async def test_a_sentence_the_model_cannot_judge_fails_closed(cfg: QuarryConfig) -> None:
    llm = FakeRewriteLLM("Rewritten pilot claim [1].", verdicts={"pilot": "malformed"})
    verified, summary = await _verify("Fine [2]. The pilot works [1].", llm, cfg)
    assert verified.sections[0].markdown == "Fine [2]."
    assert summary.unjudged == 1 and summary.dropped == 1


async def test_uncited_factual_sentences_get_a_supported_marker(cfg: QuarryConfig) -> None:
    llm = FakeRewriteLLM()
    verified, summary = await _verify(
        "Good claim [1]. The pilot stores heat at 600 degrees in large sand tanks.", llm, cfg
    )
    assert verified.sections[0].markdown == (
        "Good claim [1]. The pilot stores heat at 600 degrees in large sand tanks [1]."
    )
    assert summary.uncited_attached == 1 and summary.uncited_dropped == 0


async def test_unsupported_uncited_facts_drop_but_links_and_gap_notes_stay(
    cfg: QuarryConfig,
) -> None:
    markdown = (
        "Good claim [1]. Turning to costs. "
        "The sources reviewed did not report what the second plant will cost. "
        "Halvard Grid will open three further plants across Norway by 2031."
    )
    llm = FakeRewriteLLM(verdicts={"Halvard": "unsupported"})
    verified, summary = await _verify(markdown, llm, cfg)
    assert verified.sections[0].markdown == (
        "Good claim [1]. Turning to costs. "
        "The sources reviewed did not report what the second plant will cost."
    )
    assert summary.uncited_dropped == 1


async def test_partly_supported_uncited_fact_is_rewritten_against_its_best_chunk(
    cfg: QuarryConfig,
) -> None:
    llm = FakeRewriteLLM("The pilot stores heat at 600 degrees [1].", verdicts={"tanks": "partial"})
    verified, summary = await _verify(
        "Good claim [1]. The pilot stores heat at 600 degrees in large sand tanks.", llm, cfg
    )
    assert verified.sections[0].markdown == (
        "Good claim [1]. The pilot stores heat at 600 degrees [1]."
    )
    assert summary.uncited_attached == 1 and summary.uncited_dropped == 0


async def test_partly_supported_uncited_fact_drops_when_no_rewrite_passes(
    cfg: QuarryConfig,
) -> None:
    llm = FakeRewriteLLM("The pilot stores heat [1].", verdicts={"pilot": "partial"})
    verified, summary = await _verify(
        "Good claim [1]. The pilot stores heat at 600 degrees in large sand tanks.", llm, cfg
    )
    assert verified.sections[0].markdown == "Good claim [1]."
    assert summary.uncited_dropped == 1


async def test_uncited_fact_attaches_to_a_lower_scoring_chunk_that_supports_it(
    cfg: QuarryConfig,
) -> None:
    class SupportedByFactOne(FakeRewriteLLM):
        async def complete_typed(
            self, prompt: str, schema: Any, max_tokens: int = 512, max_retries: int = 2
        ) -> Any:
            if schema is SupportVerdict:
                self.check_prompts.append(prompt)
                verdict = "supported" if "Fact 1." in prompt or "Good" in prompt else "unsupported"
                return SupportVerdict(verdict=verdict)  # type: ignore[arg-type]
            return await super().complete_typed(prompt, schema, max_tokens, max_retries)

    verified, summary = await _verify(
        "Good claim [1]. Good claim two [2]. The pilot stores heat at 600 degrees in sand tanks.",
        SupportedByFactOne(),
        cfg,
    )
    assert verified.sections[0].markdown.endswith("in sand tanks [2].")
    assert summary.uncited_attached == 1


@pytest.mark.parametrize(
    "rewritten",
    [
        "A narrower claim [1], which then runs on past its marker",
        "A claim the model padded with a long passage copied straight out of the source text [1].",
    ],
)
async def test_rewrite_that_runs_on_or_pastes_the_source_is_rejected(
    rewritten: str, cfg: QuarryConfig
) -> None:
    llm = FakeRewriteLLM(rewritten, verdicts={"Weak": "partial"})
    verified, summary = await _verify("Good claim [1]. Weak short claim [1].", llm, cfg)
    assert verified.sections[0].markdown == "Good claim [1]."
    assert llm.calls == cfg.verify.max_rewrites
    assert summary.dropped == 1


async def test_uncited_fact_with_a_figure_the_evidence_lacks_is_dropped(cfg: QuarryConfig) -> None:
    verified, summary = await _verify(
        "Good claim [1]. The pilot stores heat at 750 degrees in large sand tanks.",
        FakeRewriteLLM(),
        cfg,
    )
    assert verified.sections[0].markdown == "Good claim [1]."
    assert summary.uncited_dropped == 1


async def test_uncited_fact_below_the_floor_never_reaches_the_model(cfg: QuarryConfig) -> None:
    llm = FakeRewriteLLM()
    scorer = FakeScorer(scores={"Halvard": -5.0})
    verified, summary = await _verify(
        "Good claim [1]. Halvard Grid will open three further plants across Norway by 2031.",
        llm,
        cfg,
        scorer,
    )
    assert verified.sections[0].markdown == "Good claim [1]."
    assert len(llm.check_prompts) == 1  # only the cited sentence was checked
    assert summary.uncited_dropped == 1


async def test_code_built_gap_sections_are_never_trimmed(cfg: QuarryConfig) -> None:
    text = "The final coverage check found every sub-question answered by the sources reviewed."
    verified, summary = await _verify(text, FakeRewriteLLM(), cfg, title=GAPS_TITLE)
    assert verified.sections[0].markdown == text
    assert summary == VerificationSummary()


async def test_the_model_sees_the_whole_cited_chunk(cfg: QuarryConfig) -> None:
    citations = CitationIndex()
    long_text = "Opening context. " * 150 + "The tail says 600 degrees."
    chunk = make_chunk(long_text)
    citations.add(chunk)
    draft = DraftReport(
        topic="t", sections=[ReportSection(title="S", markdown="Heat reaches 600 degrees [1].")]
    )
    llm = FakeRewriteLLM()
    await verify_report(
        draft,
        citations,
        {chunk.chunk_id: chunk},
        cast(Reranker, FakeScorer()),
        cast(LocalLLM, llm),
        None,
        cfg,
    )
    assert len(long_text) > 1500  # the old per-chunk cut
    assert "The tail says 600 degrees." in llm.check_prompts[0]
    assert len(long_text) < evidence_budget_chars(cfg)
