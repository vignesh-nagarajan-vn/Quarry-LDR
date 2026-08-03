"""Triage stage over a fake LocalLLM: filtering, floors, malformed drops."""

from __future__ import annotations

from typing import cast

from quarry_ldr.config import TriageSettings
from quarry_ldr.gpu.local_llm import LocalLLM
from quarry_ldr.gpu.reranker import ScoredChunk
from quarry_ldr.ingest.chunk import Chunk, make_chunk_id
from quarry_ldr.pipeline.plan import SubQuestion
from quarry_ldr.pipeline.triage import TRIAGE_PROMPT, TriageVerdict, triage_chunks


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


def make_sq() -> SubQuestion:
    return SubQuestion(
        id="sq01",
        question="What is the round-trip efficiency?",
        queries=["efficiency vesterholm", "sand battery efficiency"],
        success_criterion="An efficiency figure with a source.",
    )


class FakeLLM:
    """Maps passage substrings to verdicts; unknown passages raise ValueError
    like an exhausted-retries client."""

    def __init__(self, verdicts: dict[str, TriageVerdict]) -> None:
        self.verdicts = verdicts
        self.prompts: list[str] = []

    async def complete_typed(
        self,
        prompt: str,
        schema: type[TriageVerdict],
        max_tokens: int = 512,
        max_retries: int = 2,
    ) -> TriageVerdict:
        self.prompts.append(prompt)
        for key, verdict in self.verdicts.items():
            if key in prompt:
                return verdict
        raise ValueError("malformed")


async def test_keeps_relevant_above_floor_drops_rest() -> None:
    fake = FakeLLM(
        {
            "eff-85": TriageVerdict(
                relevant=True, claim="Efficiency is 85 percent.", evidence_span="85", confidence=0.9
            ),
            "eff-low": TriageVerdict(relevant=True, claim="c", evidence_span="e", confidence=0.1),
            "irrelevant": TriageVerdict(relevant=False, claim="", evidence_span="", confidence=0.9),
        }
    )
    scored = [
        ScoredChunk(chunk=make_chunk("passage eff-85", 0), score=3.0),
        ScoredChunk(chunk=make_chunk("passage eff-low", 1), score=2.0),
        ScoredChunk(chunk=make_chunk("passage irrelevant", 2), score=1.0),
    ]
    kept = await triage_chunks(make_sq(), scored, cast(LocalLLM, fake), TriageSettings())
    assert len(kept) == 1
    assert kept[0].verdict.claim == "Efficiency is 85 percent."
    assert kept[0].sub_question_id == "sq01"
    assert kept[0].rerank_score == 3.0


async def test_malformed_chunks_dropped_not_fatal() -> None:
    fake = FakeLLM(
        {
            "good": TriageVerdict(relevant=True, claim="c", evidence_span="e", confidence=0.8),
        }
    )
    scored = [
        ScoredChunk(chunk=make_chunk("mystery passage", 0), score=2.0),  # raises ValueError
        ScoredChunk(chunk=make_chunk("good passage", 1), score=1.0),
    ]
    kept = await triage_chunks(make_sq(), scored, cast(LocalLLM, fake), TriageSettings())
    assert len(kept) == 1
    assert kept[0].chunk.position == 1


async def test_prompt_contains_question_and_passage() -> None:
    fake = FakeLLM(
        {"the passage": TriageVerdict(relevant=True, claim="c", evidence_span="e", confidence=1.0)}
    )
    scored = [ScoredChunk(chunk=make_chunk("the passage text", 0), score=1.0)]
    await triage_chunks(make_sq(), scored, cast(LocalLLM, fake), TriageSettings())
    prompt = fake.prompts[0]
    assert "round-trip efficiency" in prompt
    assert "the passage text" in prompt


async def test_confidence_floor_is_configurable() -> None:
    fake = FakeLLM(
        {"p": TriageVerdict(relevant=True, claim="c", evidence_span="e", confidence=0.4)}
    )
    scored = [ScoredChunk(chunk=make_chunk("p", 0), score=1.0)]
    kept_default = await triage_chunks(make_sq(), scored, cast(LocalLLM, fake), TriageSettings())
    kept_strict = await triage_chunks(
        make_sq(), scored, cast(LocalLLM, fake), TriageSettings(confidence_floor=0.5)
    )
    assert len(kept_default) == 1
    assert len(kept_strict) == 0


async def test_empty_input() -> None:
    fake = FakeLLM({})
    kept = await triage_chunks(make_sq(), [], cast(LocalLLM, fake), TriageSettings())
    assert kept == []


def test_prompt_template_shape() -> None:
    rendered = TRIAGE_PROMPT.format(question="Q?", passage="P.")
    assert '"relevant"' in rendered
    assert "Q?" in rendered and "P." in rendered
    assert "JSON" in rendered


async def test_long_passage_truncated_in_prompt() -> None:
    fake = FakeLLM(
        {"start-marker": TriageVerdict(relevant=True, claim="c", evidence_span="e", confidence=1.0)}
    )
    long_text = "start-marker " + "x" * 10_000
    scored = [ScoredChunk(chunk=make_chunk(long_text, 0), score=1.0)]
    await triage_chunks(make_sq(), scored, cast(LocalLLM, fake), TriageSettings())
    assert len(fake.prompts[0]) < 8_000
