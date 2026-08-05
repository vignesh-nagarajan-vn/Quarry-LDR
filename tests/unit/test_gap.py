"""Gap analysis: digest content, typed parsing, ledger iteration tagging."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx
from anthropic import AsyncAnthropic

from quarry_ldr.config import QuarryConfig
from quarry_ldr.ledger import Ledger
from quarry_ldr.pipeline.gap import GAP_SYSTEM, analyze_gaps, coverage_digest

# Reuse the local factories from the synthesize tests' pattern.
from quarry_ldr.pipeline.plan import ResearchPlan, SubQuestion
from quarry_ldr.pipeline.triage import TriagedChunk, TriageVerdict
from quarry_ldr.providers.anthropic_client import AnthropicProvider

MESSAGES = "https://api.anthropic.com/v1/messages"


def make_plan_obj() -> ResearchPlan:
    return ResearchPlan(
        topic="sand battery",
        sub_questions=[
            SubQuestion(
                id=f"sq{i:02d}",
                question=f"Question {i}?",
                queries=[f"q{i}a", f"q{i}b"],
                success_criterion=f"Criterion {i} is met.",
            )
            for i in range(1, 9)
        ],
    )


def test_coverage_digest_lists_criteria_and_claims() -> None:
    from quarry_ldr.ingest.chunk import Chunk, make_chunk_id

    url = "https://vesterholm-times.example/2024/a"
    chunk = Chunk(
        chunk_id=make_chunk_id(url, 0),
        url=url,
        doc_title="d",
        heading_path=[],
        text="t",
        token_count=1,
        position=0,
        start_char=0,
        end_char=1,
    )
    evidence = [
        TriagedChunk(
            sub_question_id="sq01",
            chunk=chunk,
            verdict=TriageVerdict(
                relevant=True, claim="Efficiency is 85 percent.", evidence_span="85", confidence=1.0
            ),
            rerank_score=1.0,
        )
    ]
    digest = coverage_digest(make_plan_obj(), evidence)
    assert "sq01: Question 1?" in digest
    assert "success criterion: Criterion 1 is met." in digest
    assert "claim: Efficiency is 85 percent." in digest
    assert "(no evidence collected)" in digest  # sq02..sq08 have none
    assert GAP_SYSTEM.startswith("You audit")


async def test_analyze_gaps_parses_and_tags_iteration(
    cfg: QuarryConfig, fixtures_dir: Path
) -> None:
    body = json.loads((fixtures_dir / "anthropic" / "gap_response.json").read_text("utf-8"))
    ledger = Ledger()
    provider = AnthropicProvider(
        cfg, ledger, client=AsyncAnthropic(api_key="test-key-not-real", max_retries=0)
    )
    with respx.mock:
        route = respx.post(MESSAGES).mock(return_value=httpx.Response(200, json=body))
        gap = await analyze_gaps(make_plan_obj(), [], provider, cfg, iteration=2)
        payload = json.loads(route.calls[0].request.content)
    assert gap.saturated is False
    assert gap.new_queries
    assert payload["model"] == "claude-sonnet-5"
    assert "Criterion 3 is met." in payload["messages"][0]["content"]
    assert "iterations so far: 3" in payload["messages"][0]["content"]
    entry = ledger.entries[0]
    assert entry.stage == "gap"
    assert entry.iteration == 2


def test_coverage_assessment_accepts_null_missing() -> None:
    """Haiku emits an explicit null for covered sub-questions where Sonnet
    emits ""; both must validate without a schema-retry round trip."""
    from quarry_ldr.pipeline.gap import CoverageAssessment

    parsed = CoverageAssessment.model_validate(
        {"sub_question_id": "sq01", "covered": True, "missing": None}
    )
    assert parsed.missing is None
    assert (
        CoverageAssessment.model_validate({"sub_question_id": "sq02", "covered": False}).missing
        == ""
    )


async def test_analyze_gaps_model_override(cfg: QuarryConfig, fixtures_dir: Path) -> None:
    """The assisted engine passes model=; default call sites keep cfg.models.gap."""
    body = json.loads((fixtures_dir / "anthropic" / "gap_response.json").read_text("utf-8"))
    provider = AnthropicProvider(
        cfg, Ledger(), client=AsyncAnthropic(api_key="test-key-not-real", max_retries=0)
    )
    with respx.mock:
        route = respx.post(MESSAGES).mock(return_value=httpx.Response(200, json=body))
        await analyze_gaps(
            make_plan_obj(), [], provider, cfg, iteration=0, model="claude-haiku-4-5-20251001"
        )
        payload = json.loads(route.calls[0].request.content)
    assert payload["model"] == "claude-haiku-4-5-20251001"


def _bulk_evidence(n_per_sq: int = 40, claim_chars: int = 120) -> list:
    from quarry_ldr.ingest.chunk import Chunk, make_chunk_id

    url = "https://vesterholm-times.example/2024/a"
    chunk = Chunk(
        chunk_id=make_chunk_id(url, 0),
        url=url,
        doc_title="d",
        heading_path=[],
        text="t",
        token_count=1,
        position=0,
        start_char=0,
        end_char=1,
    )
    return [
        TriagedChunk(
            sub_question_id=f"sq{sq:02d}",
            chunk=chunk,
            verdict=TriageVerdict(
                relevant=True,
                claim=f"Claim {sq}-{i}: " + "x" * claim_chars,
                evidence_span="s",
                confidence=1.0,
            ),
            rerank_score=1.0,
        )
        for sq in range(1, 9)
        for i in range(n_per_sq)
    ]


def test_budgeted_digest_shrinks_to_fit() -> None:
    from quarry_ldr.ingest.chunk import HeuristicTokenCounter
    from quarry_ldr.pipeline.gap import budgeted_digest

    plan = make_plan_obj()
    evidence = _bulk_evidence()
    counter = HeuristicTokenCounter()
    full = coverage_digest(plan, evidence)
    budget = counter.count(full) // 4
    digest = budgeted_digest(plan, evidence, budget)
    assert counter.count(digest) <= budget
    assert digest.count("claim:") < full.count("claim:")
    # Every sub-question still appears; coverage shrinks evenly, never drops.
    for sq in range(1, 9):
        assert f"sq{sq:02d}:" in digest


def test_budgeted_digest_last_resort_truncates_claims() -> None:
    from quarry_ldr.pipeline.gap import budgeted_digest

    plan = make_plan_obj()
    evidence = _bulk_evidence(claim_chars=4000)
    digest = budgeted_digest(plan, evidence, budget_tokens=50)
    # One claim per sub-question, hard-truncated: no 4000-char lines survive.
    assert digest.count("claim:") == 8
    assert max(len(line) for line in digest.splitlines()) < 300


async def test_analyze_gaps_digest_budget_bounds_the_prompt(cfg: QuarryConfig) -> None:
    from typing import Any

    from quarry_ldr.ingest.chunk import HeuristicTokenCounter

    captured: dict[str, Any] = {}

    class CapturingProvider:
        async def complete_typed(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return kwargs["schema"].model_validate(
                {"saturated": True, "assessments": [], "new_queries": [], "rationale": ""}
            )

    gap = await analyze_gaps(
        make_plan_obj(),
        _bulk_evidence(),
        CapturingProvider(),  # type: ignore[arg-type]
        cfg,
        iteration=0,
        digest_budget_tokens=400,
        max_tokens=1200,
    )
    assert gap.saturated is True
    assert captured["max_tokens"] == 1200
    # Digest budget plus the small topic/iteration header.
    assert HeuristicTokenCounter().count(captured["prompt"]) <= 400 + 60
