"""End-to-end single pass on the fixture corpus: plan through cited report.

CPU only, no network (respx serves the corpus), no API key (fake provider),
fake GPU components. This is the M8 gate: the report exists, every citation
resolves, the run store holds every stage, and resume replays completed
stages without recomputation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
import pytest
import respx
from numpy.typing import NDArray

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.embedder import Embedder
from quarry_ldr.gpu.local_llm import LocalLLM
from quarry_ldr.gpu.reranker import Reranker, ScoredChunk
from quarry_ldr.ingest.chunk import Chunk
from quarry_ldr.ingest.fetch import Fetcher
from quarry_ldr.ingest.search import SearchResult, SearxClient
from quarry_ldr.ledger import TokenUsage
from quarry_ldr.pipeline.plan import _PlanPayload
from quarry_ldr.pipeline.run import Orchestrator
from quarry_ldr.pipeline.triage import TriageVerdict
from quarry_ldr.providers.anthropic_client import CompletionResult
from quarry_ldr.state import RunStatus, RunStore, Stage, StageStatus

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _manifest() -> list[dict[str, Any]]:
    return json.loads((FIXTURES / "manifest.json").read_text("utf-8"))["docs"]


def _hash_vector(text: str, dim: int = 32) -> NDArray[np.float32]:
    digest = hashlib.sha256(text.lower().encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    vector = rng.standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


class FakeSearx:
    def __init__(self) -> None:
        self.calls = 0

    async def search_many(self, queries: list[str], num_results: int = 10) -> list[SearchResult]:
        self.calls += 1
        return [
            SearchResult(url=doc["url"], title=doc["title"], snippet="", query=queries[0])
            for doc in _manifest()
        ]


class FakeEmbedder:
    dim = 32

    async def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([_hash_vector(text, self.dim) for text in texts])

    async def embed_query(self, text: str) -> NDArray[np.float32]:
        return _hash_vector(text, self.dim)


class FakeReranker:
    async def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[ScoredChunk]:
        query_words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 3}

        def score(chunk: Chunk) -> float:
            words = set(re.findall(r"\w+", chunk.text.lower()))
            return float(len(query_words & words))

        scored = sorted(
            (ScoredChunk(chunk=chunk, score=score(chunk)) for chunk in chunks),
            key=lambda s: -s.score,
        )
        return scored[:top_k]


class FakeLocalLLM:
    async def complete_typed(
        self, prompt: str, schema: type[TriageVerdict], max_tokens: int = 512, max_retries: int = 2
    ) -> TriageVerdict:
        relevant = "sand" in prompt.lower() or "vesterholm" in prompt.lower()
        return TriageVerdict(
            relevant=relevant,
            claim="A fixture claim about the pilot." if relevant else "",
            evidence_span="fixture span" if relevant else "",
            confidence=0.8 if relevant else 0.1,
        )


class FakeProvider:
    """Plan + cached synthesis without an API key; records usage in a ledger
    when given one so the report's cost table is exercised."""

    def __init__(self, fail_on_synthesize: bool = False) -> None:
        self.fail_on_synthesize = fail_on_synthesize
        self.synth_calls = 0
        self.corpora: list[str] = []

    async def complete_typed(self, **kwargs: Any) -> Any:
        schema = kwargs["schema"]
        assert schema is _PlanPayload
        return _PlanPayload.model_validate(
            {
                "sub_questions": [
                    {
                        "id": f"sq{i:02d}",
                        "question": f"Fixture question {i} about the Vesterholm sand battery?",
                        "queries": [f"vesterholm sand battery {i}", f"sand storage pilot {i}"],
                        "success_criterion": f"Criterion {i}.",
                    }
                    for i in range(1, 9)
                ]
            }
        )

    async def complete_with_cached_corpus(self, **kwargs: Any) -> CompletionResult:
        if self.fail_on_synthesize:
            raise RuntimeError("synthetic crash during synthesis")
        self.synth_calls += 1
        corpus: str = kwargs["corpus"]
        self.corpora.append(corpus)
        numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", corpus)})[:2]
        cites = " and ".join(f"[{n}]" for n in numbers) if numbers else ""
        return CompletionResult(
            text=f"Fixture section text citing {cites}.",
            usage=TokenUsage(input_tokens=100, output_tokens=50),
            model=kwargs["model"],
            stop_reason="end_turn",
        )


def _mock_corpus_routes() -> None:
    """Register respx routes for every fixture URL and per-domain robots."""
    by_url = {doc["url"]: doc for doc in _manifest()}
    domains = {doc["domain"] for doc in _manifest()}
    for domain in domains:
        robots_path = FIXTURES / "robots" / f"{domain}.txt"
        respx.get(f"https://{domain}/robots.txt").mock(
            return_value=httpx.Response(200, text=robots_path.read_text("utf-8"))
        )
    for url, doc in by_url.items():
        body = (FIXTURES / "html" / doc["file"]).read_bytes()
        respx.get(url).mock(
            return_value=httpx.Response(
                200, content=body, headers={"content-type": "text/html; charset=utf-8"}
            )
        )


def _orchestrator(cfg: QuarryConfig, provider: FakeProvider) -> Orchestrator:
    # Politeness is covered by fetch tests; fixture runs should not sleep.
    cfg.fetch.per_domain_rps = 500.0
    return Orchestrator(
        cfg,
        searx=cast(SearxClient, FakeSearx()),
        fetcher=Fetcher(cfg.fetch),
        embedder=cast(Embedder, FakeEmbedder()),
        reranker=cast(Reranker, FakeReranker()),
        local_llm=cast(LocalLLM, FakeLocalLLM()),
        provider=provider,
    )


async def test_single_pass_produces_cited_report(cfg: QuarryConfig) -> None:
    provider = FakeProvider()
    with respx.mock:
        _mock_corpus_routes()
        result = await _orchestrator(cfg, provider).research("the vesterholm sand battery pilot")

    report_path = Path(result.report_path)
    assert report_path.is_file()
    report = report_path.read_text(encoding="utf-8")

    # Cited sections, references resolving to fixture URLs, ledger, manifest.
    assert report.startswith("# the vesterholm sand battery pilot")
    assert "## Overview" in report and "## Conclusions" in report
    assert re.search(r"citing \[\d+\]", report)
    assert "## References" in report
    assert ".example/" in report
    assert "## Cost ledger" in report
    assert "## Run manifest" in report

    # Pipeline actually compressed: robots-disallowed and empty docs dropped,
    # near-duplicates deduplicated.
    assert result.n_sources < 40
    assert result.n_chunks_indexed > 0
    assert result.n_chunks_evidence > 0
    assert provider.synth_calls == 10
    assert len(set(provider.corpora)) == 1

    # Run store: every stage completed.
    async with RunStore(cfg.run.data_dir / "runs.db") as store:
        run = await store.get_run(result.run_id)
        assert run.status is RunStatus.COMPLETED
        stages = await store.stages(result.run_id)
        assert {record.stage for record in stages} == {
            Stage.PLAN,
            Stage.SEARCH,
            Stage.FETCH,
            Stage.EXTRACT,
            Stage.CHUNK,
            Stage.EMBED,
            Stage.DEDUP,
            Stage.INDEX,
            Stage.RETRIEVE,
            Stage.RERANK,
            Stage.TRIAGE,
            Stage.SYNTHESIZE,
            Stage.RENDER,
        }
        assert all(record.status is StageStatus.COMPLETED for record in stages)


async def test_robots_disallowed_url_never_fetched(cfg: QuarryConfig) -> None:
    disallowed = next(d for d in _manifest() if d["kind"] == "robots_disallowed")
    with respx.mock:
        _mock_corpus_routes()
        page_route = respx.get(disallowed["url"])
        await _orchestrator(cfg, FakeProvider()).research("sand battery")
        assert page_route.call_count == 0


async def test_crash_then_resume_replays_without_recompute(cfg: QuarryConfig) -> None:
    crashing = FakeProvider(fail_on_synthesize=True)
    with respx.mock:
        _mock_corpus_routes()
        orchestrator = _orchestrator(cfg, crashing)
        with pytest.raises(RuntimeError, match="synthetic crash"):
            await orchestrator.research("sand battery resume case")

        # Failed run: stages up to TRIAGE completed, SYNTHESIZE failed.
        async with RunStore(cfg.run.data_dir / "runs.db") as store:
            runs = await store.list_runs()
            run_id = runs[0].run_id
            assert runs[0].status is RunStatus.FAILED
            triage = await store.get_stage(run_id, Stage.TRIAGE)
            synth = await store.get_stage(run_id, Stage.SYNTHESIZE)
            assert triage is not None and triage.status is StageStatus.COMPLETED
            assert synth is not None and synth.status is StageStatus.FAILED
            before = {record.stage: record.started_at for record in await store.stages(run_id)}

        working = FakeProvider()
        result = await _orchestrator(cfg, working).resume(run_id)

        assert Path(result.report_path).is_file()
        async with RunStore(cfg.run.data_dir / "runs.db") as store:
            assert (await store.get_run(run_id)).status is RunStatus.COMPLETED
            after = {record.stage: record.started_at for record in await store.stages(run_id)}
        # Completed stages were replayed, not re-run: identical start stamps.
        for stage in (Stage.PLAN, Stage.SEARCH, Stage.FETCH, Stage.TRIAGE):
            assert after[stage] == before[stage]
        # The failed stage was actually re-executed.
        assert after[Stage.SYNTHESIZE] != before[Stage.SYNTHESIZE]


async def test_inspect_summarizes_stages(cfg: QuarryConfig) -> None:
    with respx.mock:
        _mock_corpus_routes()
        orchestrator = _orchestrator(cfg, FakeProvider())
        result = await orchestrator.research("inspect me")
        state = await orchestrator.inspect(result.run_id)
    assert state["run"]["status"] == "completed"
    stage_names = [record["stage"] for record in state["stages"]]
    assert stage_names[0] == "plan" and stage_names[-1] == "render"
    assert all("payload_keys" in record for record in state["stages"])
