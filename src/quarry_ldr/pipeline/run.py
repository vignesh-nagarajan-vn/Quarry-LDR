"""The orchestrator: a plain async state machine over the SQLite run store.

Every stage transition is a row; ``resume`` re-enters the same execution path
and completed stages return their persisted payloads instantly, so nothing is
recomputed; a crash mid-fetch loses nothing because fetched bodies live in
the disk cache keyed by URL.

GPU stages are batched by model, never interleaved: all embedding (queries
included), then one swap to the reranker, then one swap to the local LLM.

Single pass (M8); the gap-analysis loop lands in M9.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.arbiter import GpuBackend, VramArbiter
from quarry_ldr.gpu.embedder import Embedder
from quarry_ldr.gpu.local_llm import LlamaServer, LocalLLM
from quarry_ldr.gpu.reranker import Reranker, ScoredChunk
from quarry_ldr.index.store import VectorStore
from quarry_ldr.ingest.chunk import Chunk, chunk_document
from quarry_ldr.ingest.dedup import dedup_chunks
from quarry_ldr.ingest.extract import extract_document
from quarry_ldr.ingest.fetch import Fetcher
from quarry_ldr.ingest.search import SearxClient
from quarry_ldr.ledger import Ledger
from quarry_ldr.logging import get_logger, setup_logging
from quarry_ldr.pipeline.plan import ResearchPlan, make_plan
from quarry_ldr.pipeline.retrieve import rerank_candidates, retrieve_candidates
from quarry_ldr.pipeline.synthesize import DraftReport, build_evidence_corpus, synthesize
from quarry_ldr.pipeline.triage import TriagedChunk, triage_chunks
from quarry_ldr.providers.anthropic_client import AnthropicProvider
from quarry_ldr.report.citations import CitationIndex
from quarry_ldr.report.render import RunManifest, render_report, write_report
from quarry_ldr.state import RunStatus, RunStore, Stage

logger = get_logger(component="orchestrator")


class RunResult(BaseModel):
    run_id: str
    report_path: str
    total_cost_usd: float
    iterations: int
    n_sources: int
    n_chunks_indexed: int
    n_chunks_evidence: int


def _default_gpu_backend() -> GpuBackend | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    from quarry_ldr.gpu.arbiter import TorchCudaBackend

    return TorchCudaBackend()


class Orchestrator:
    """Owns component construction, stage sequencing, checkpointing, and the loop.

    Components are constructed lazily and are injectable via keyword
    ``overrides`` for tests: store, arbiter, searx, fetcher, embedder,
    reranker, llama_server, local_llm, provider, vstore.
    """

    def __init__(self, cfg: QuarryConfig, **overrides: Any) -> None:
        self.cfg = cfg
        self._ov = overrides

    # ------------------------------------------------------------- plumbing

    def _data_dir(self) -> Path:
        return Path(self.cfg.run.data_dir)

    def _store(self) -> RunStore:
        store: RunStore = self._ov.get("store") or RunStore(self._data_dir() / "runs.db")
        return store

    async def _stage(
        self,
        store: RunStore,
        run_id: str,
        stage: Stage,
        iteration: int,
        compute: Callable[[], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Run a stage with checkpointing: a completed stage replays its
        payload without recomputation, which is exactly what resume does."""
        existing = await store.get_stage(run_id, stage, iteration)
        if existing is not None and existing.status.value == "completed":
            logger.info("stage_replayed", stage=stage.value, iteration=iteration)
            return existing.payload
        await store.start_stage(run_id, stage, iteration)
        import structlog

        structlog.contextvars.bind_contextvars(stage=stage.value)
        try:
            payload = await compute()
        except Exception as exc:
            await store.fail_stage(run_id, stage, f"{type(exc).__name__}: {exc}", iteration)
            raise
        finally:
            structlog.contextvars.unbind_contextvars("stage")
        await store.complete_stage(run_id, stage, payload, iteration)
        return payload

    async def _persist_new_ledger_entries(
        self, store: RunStore, run_id: str, ledger: Ledger, persisted: int
    ) -> int:
        entries = ledger.dump()
        for entry in entries[persisted:]:
            await store.append_ledger_entry(run_id, entry)
        return len(entries)

    # ------------------------------------------------------------ public api

    async def research(self, topic: str) -> RunResult:
        """Create a run and execute PLAN through RENDER."""
        setup_logging(log_dir=self._data_dir() / "logs")
        async with self._store() as store:
            run = await store.create_run(topic, self.cfg.snapshot())
            import structlog

            structlog.contextvars.bind_contextvars(run_id=run.run_id)
            logger.info("run_created", topic=topic)
            return await self._drive(store, run.run_id, topic, Ledger(self.cfg.run.cost_cap_usd))

    async def resume(self, run_id: str) -> RunResult:
        """Continue an interrupted run from its last completed stage."""
        setup_logging(log_dir=self._data_dir() / "logs", run_id=run_id)
        async with self._store() as store:
            run = await store.get_run(run_id)
            ledger = Ledger.load(
                await store.ledger_entries(run_id), cost_cap_usd=self.cfg.run.cost_cap_usd
            )
            logger.info("run_resumed", topic=run.topic)
            return await self._drive(store, run_id, run.topic, ledger)

    async def inspect(self, run_id: str) -> dict[str, Any]:
        """Stage-by-stage state as JSON-safe data for `quarry inspect`."""
        async with self._store() as store:
            run = await store.get_run(run_id)
            stages = await store.stages(run_id)
            entries = await store.ledger_entries(run_id)
            return {
                "run": run.model_dump(mode="json", exclude={"config_snapshot"}),
                "stages": [
                    {
                        **record.model_dump(mode="json", exclude={"payload"}),
                        "payload_keys": sorted(record.payload),
                        "payload_summary": {
                            key: (f"[{len(value)} items]" if isinstance(value, list) else value)
                            for key, value in record.payload.items()
                            if not isinstance(value, dict)
                        },
                    }
                    for record in stages
                ],
                "ledger": entries,
            }

    # ------------------------------------------------------------- the drive

    async def _drive(self, store: RunStore, run_id: str, topic: str, ledger: Ledger) -> RunResult:
        started_at = datetime.now(UTC)
        await store.set_run_status(run_id, RunStatus.RUNNING)
        persisted = len(await store.ledger_entries(run_id))

        arbiter: VramArbiter = self._ov.get("arbiter") or VramArbiter(
            self.cfg.gpu.vram_budget_mb, _default_gpu_backend()
        )
        provider: AnthropicProvider = self._ov.get("provider") or AnthropicProvider(
            self.cfg, ledger
        )
        embedder: Embedder = self._ov.get("embedder") or Embedder(self.cfg, arbiter)
        reranker: Reranker = self._ov.get("reranker") or Reranker(self.cfg, arbiter)
        fetcher: Fetcher = self._ov.get("fetcher") or Fetcher(self.cfg.fetch)
        searx: SearxClient = self._ov.get("searx") or SearxClient(
            self.cfg.search.searxng_url, self.cfg.search.timeout_s
        )
        vstore: VectorStore = self._ov.get("vstore") or VectorStore(
            self._data_dir() / "index" / run_id, embedder.dim
        )
        llama_server: LlamaServer | None = self._ov.get("llama_server")
        local_llm: LocalLLM | None = self._ov.get("local_llm")
        owns_llama = llama_server is None and local_llm is None

        try:
            # ---- PLAN
            plan_payload = await self._stage(
                store,
                run_id,
                Stage.PLAN,
                0,
                lambda: self._compute_plan(topic, provider),
            )
            plan = ResearchPlan.model_validate(plan_payload)
            persisted = await self._persist_new_ledger_entries(store, run_id, ledger, persisted)

            iteration = 0
            queries = plan.all_queries()

            # ---- SEARCH
            search_payload = await self._stage(
                store,
                run_id,
                Stage.SEARCH,
                iteration,
                lambda: self._compute_search(searx, queries),
            )
            urls: list[str] = search_payload["urls"]

            # ---- FETCH
            fetch_payload = await self._stage(
                store,
                run_id,
                Stage.FETCH,
                iteration,
                lambda: self._compute_fetch(fetcher, urls),
            )
            ok_urls: list[str] = fetch_payload["ok_urls"]

            # ---- EXTRACT + CHUNK
            extract_payload = await self._stage(
                store,
                run_id,
                Stage.EXTRACT,
                iteration,
                lambda: self._compute_extract(fetcher, ok_urls),
            )
            chunk_payload = await self._stage(
                store,
                run_id,
                Stage.CHUNK,
                iteration,
                lambda: self._compute_chunk(fetcher, extract_payload["extracted_urls"]),
            )
            chunks = [Chunk.model_validate(raw) for raw in chunk_payload["chunks"]]

            # ---- EMBED (all texts, one residency)
            embed_payload = await self._stage(
                store,
                run_id,
                Stage.EMBED,
                iteration,
                lambda: self._compute_embed(embedder, chunks, run_id, iteration),
            )
            embeddings = np.load(embed_payload["path"])["embeddings"]

            # ---- DEDUP
            dedup_payload = await self._stage(
                store,
                run_id,
                Stage.DEDUP,
                iteration,
                lambda: self._compute_dedup(chunks, embeddings),
            )
            kept_indices: list[int] = dedup_payload["kept"]
            kept_chunks = [chunks[i] for i in kept_indices]
            kept_embeddings = embeddings[kept_indices] if len(kept_indices) else embeddings[:0]

            # ---- INDEX
            index_payload = await self._stage(
                store,
                run_id,
                Stage.INDEX,
                iteration,
                lambda: self._compute_index(vstore, kept_chunks, kept_embeddings),
            )

            # ---- RETRIEVE (embedder resident), then RERANK (one swap)
            retrieve_payload = await self._stage(
                store,
                run_id,
                Stage.RETRIEVE,
                iteration,
                lambda: self._compute_retrieve(plan, vstore, embedder),
            )
            rerank_payload = await self._stage(
                store,
                run_id,
                Stage.RERANK,
                iteration,
                lambda: self._compute_rerank(plan, retrieve_payload, vstore, reranker),
            )

            # ---- TRIAGE (one swap to the local LLM)
            if owns_llama:
                llama_server = LlamaServer(self.cfg, arbiter, Path(self.cfg.run.models_dir))
                local_llm = LocalLLM(llama_server.base_url)
            triage_payload = await self._stage(
                store,
                run_id,
                Stage.TRIAGE,
                iteration,
                lambda: self._compute_triage(plan, rerank_payload, vstore, llama_server, local_llm),
            )
            evidence = [TriagedChunk.model_validate(raw) for raw in triage_payload["evidence"]]

            # ---- SYNTHESIZE (section by section over the cached corpus)
            citations = CitationIndex()
            synth_payload = await self._stage(
                store,
                run_id,
                Stage.SYNTHESIZE,
                0,
                lambda: self._compute_synthesize(plan, evidence, provider, citations),
            )
            persisted = await self._persist_new_ledger_entries(store, run_id, ledger, persisted)
            draft = DraftReport.model_validate(synth_payload["draft"])
            if len(citations) == 0:
                # Resumed past synthesis: rebuild deterministic numbering.
                build_evidence_corpus(evidence, citations)

            # ---- RENDER
            render_payload = await self._stage(
                store,
                run_id,
                Stage.RENDER,
                0,
                lambda: self._compute_render(
                    run_id,
                    topic,
                    draft,
                    citations,
                    ledger,
                    started_at,
                    iteration + 1,
                    {
                        "n_queries": len(queries),
                        "n_urls_fetched": len(urls),
                        "n_docs_extracted": len(extract_payload["extracted_urls"]),
                        "n_chunks": len(chunks),
                        "n_chunks_after_dedup": len(kept_chunks),
                        "n_chunks_evidence": len(evidence),
                    },
                ),
            )
            await store.set_run_status(run_id, RunStatus.COMPLETED)
            await self._persist_new_ledger_entries(store, run_id, ledger, persisted)
            return RunResult(
                run_id=run_id,
                report_path=render_payload["report_path"],
                total_cost_usd=ledger.total_cost_usd,
                iterations=iteration + 1,
                n_sources=len(ok_urls),
                n_chunks_indexed=index_payload["n_total"],
                n_chunks_evidence=len(evidence),
            )
        except Exception:
            await store.set_run_status(run_id, RunStatus.FAILED)
            raise
        finally:
            if owns_llama and llama_server is not None:
                await llama_server.stop()
            await fetcher.aclose()

    # ------------------------------------------------------- stage computers

    async def _compute_plan(self, topic: str, provider: AnthropicProvider) -> dict[str, Any]:
        plan = await make_plan(topic, provider, self.cfg)
        return plan.model_dump(mode="json")

    async def _compute_search(self, searx: SearxClient, queries: list[str]) -> dict[str, Any]:
        results = await searx.search_many(queries, self.cfg.search.results_per_query)
        urls = [result.url for result in results]
        logger.info("search_done", n_queries=len(queries), n_urls=len(urls))
        return {"urls": urls, "n_queries": len(queries)}

    async def _compute_fetch(self, fetcher: Fetcher, urls: list[str]) -> dict[str, Any]:
        results = await fetcher.fetch_many(urls)
        ok_urls = [result.url for result in results if result.ok]
        statuses: dict[str, int] = {}
        for result in results:
            statuses[result.status.value] = statuses.get(result.status.value, 0) + 1
        logger.info("fetch_done", n_ok=len(ok_urls), statuses=statuses)
        return {"ok_urls": ok_urls, "statuses": statuses}

    async def _compute_extract(self, fetcher: Fetcher, ok_urls: list[str]) -> dict[str, Any]:
        extracted_urls: list[str] = []
        for url in ok_urls:
            result = await fetcher.fetch(url)  # cache hit by construction
            if not result.ok:
                continue
            doc = extract_document(result.read_bytes(), url)
            if doc is not None and doc.blocks:
                extracted_urls.append(url)
        logger.info(
            "extract_done",
            n_docs=len(extracted_urls),
            n_rejected=len(ok_urls) - len(extracted_urls),
        )
        return {"extracted_urls": extracted_urls}

    async def _compute_chunk(self, fetcher: Fetcher, extracted_urls: list[str]) -> dict[str, Any]:
        chunks: list[Chunk] = []
        for url in extracted_urls:
            result = await fetcher.fetch(url)
            doc = extract_document(result.read_bytes(), url)
            if doc is None:
                continue
            chunks.extend(chunk_document(doc, self.cfg.chunk))
        logger.info("chunk_done", n_chunks=len(chunks))
        return {"chunks": [chunk.model_dump(mode="json") for chunk in chunks]}

    async def _compute_embed(
        self, embedder: Embedder, chunks: list[Chunk], run_id: str, iteration: int
    ) -> dict[str, Any]:
        embeddings = await embedder.embed_texts([chunk.text for chunk in chunks])
        out_dir = self._data_dir() / "runs" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"embeddings-i{iteration}.npz"
        np.savez_compressed(path, embeddings=embeddings)
        logger.info("embed_done", n=len(chunks), dim=embedder.dim)
        return {"path": str(path), "n": len(chunks)}

    async def _compute_dedup(self, chunks: list[Chunk], embeddings: np.ndarray) -> dict[str, Any]:
        result = dedup_chunks(chunks, embeddings.astype(np.float32), self.cfg.dedup)
        logger.info(
            "dedup_done",
            n_input=result.n_input,
            n_kept=result.n_kept,
            drop_rate=round(result.drop_rate, 3),
        )
        return {
            "kept": result.kept,
            "dropped": {str(k): v for k, v in result.dropped.items()},
            "drop_rate": result.drop_rate,
        }

    async def _compute_index(
        self, vstore: VectorStore, chunks: list[Chunk], embeddings: np.ndarray
    ) -> dict[str, Any]:
        def _add() -> int:
            vstore.open()
            return vstore.add(chunks, embeddings.astype(np.float32))

        n_inserted = await asyncio.to_thread(_add)
        n_total = await asyncio.to_thread(vstore.count)
        logger.info("index_done", n_inserted=n_inserted, n_total=n_total)
        return {"n_inserted": n_inserted, "n_total": n_total}

    async def _compute_retrieve(
        self, plan: ResearchPlan, vstore: VectorStore, embedder: Embedder
    ) -> dict[str, Any]:
        await asyncio.to_thread(vstore.open)
        per_sq: dict[str, list[list[Any]]] = {}
        for sub_question in plan.sub_questions:
            candidates = await retrieve_candidates(
                sub_question, vstore, embedder, self.cfg.retrieve
            )
            per_sq[sub_question.id] = [[c.chunk.chunk_id, c.score] for c in candidates]
        logger.info("retrieve_done", n_sq=len(per_sq))
        return {"per_sq": per_sq}

    async def _compute_rerank(
        self,
        plan: ResearchPlan,
        retrieve_payload: dict[str, Any],
        vstore: VectorStore,
        reranker: Reranker,
    ) -> dict[str, Any]:
        await asyncio.to_thread(vstore.open)
        all_ids = sorted({cid for rows in retrieve_payload["per_sq"].values() for cid, _ in rows})
        lookup = await asyncio.to_thread(vstore.get_many, all_ids)
        per_sq: dict[str, list[list[Any]]] = {}
        for sub_question in plan.sub_questions:
            rows = retrieve_payload["per_sq"].get(sub_question.id, [])
            candidates = [
                ScoredChunk(chunk=lookup[cid], score=score) for cid, score in rows if cid in lookup
            ]
            reranked = await rerank_candidates(
                sub_question, candidates, reranker, self.cfg.retrieve
            )
            per_sq[sub_question.id] = [[c.chunk.chunk_id, c.score] for c in reranked]
        logger.info("rerank_done", n_sq=len(per_sq))
        return {"per_sq": per_sq}

    async def _compute_triage(
        self,
        plan: ResearchPlan,
        rerank_payload: dict[str, Any],
        vstore: VectorStore,
        llama_server: LlamaServer | None,
        local_llm: LocalLLM | None,
    ) -> dict[str, Any]:
        assert local_llm is not None
        if llama_server is not None:
            await llama_server.start()
        await asyncio.to_thread(vstore.open)
        all_ids = sorted({cid for rows in rerank_payload["per_sq"].values() for cid, _ in rows})
        lookup = await asyncio.to_thread(vstore.get_many, all_ids)
        evidence: list[TriagedChunk] = []
        for sub_question in plan.sub_questions:
            rows = rerank_payload["per_sq"].get(sub_question.id, [])
            scored = [
                ScoredChunk(chunk=lookup[cid], score=score) for cid, score in rows if cid in lookup
            ]
            evidence.extend(await triage_chunks(sub_question, scored, local_llm, self.cfg.triage))
        logger.info("triage_done_all", n_evidence=len(evidence))
        return {"evidence": [item.model_dump(mode="json") for item in evidence]}

    async def _compute_synthesize(
        self,
        plan: ResearchPlan,
        evidence: list[TriagedChunk],
        provider: AnthropicProvider,
        citations: CitationIndex,
    ) -> dict[str, Any]:
        draft = await synthesize(plan, evidence, provider, self.cfg, citations)
        logger.info("synthesize_done", n_sections=len(draft.sections))
        return {"draft": draft.model_dump(mode="json"), "corpus_hash": draft.corpus_hash}

    async def _compute_render(
        self,
        run_id: str,
        topic: str,
        draft: DraftReport,
        citations: CitationIndex,
        ledger: Ledger,
        started_at: datetime,
        iterations: int,
        counts: dict[str, int],
    ) -> dict[str, Any]:
        manifest = RunManifest(
            run_id=run_id,
            topic=topic,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            iterations=iterations,
            models={
                "plan": self.cfg.models.plan,
                "gap": self.cfg.models.gap,
                "synthesize": self.cfg.models.synthesize,
                "embedder": self.cfg.models.embedder,
                "reranker": self.cfg.models.reranker,
                "triage": self.cfg.models.triage_gguf_file,
            },
            config_snapshot=self.cfg.snapshot(),
            **counts,
        )
        markdown = render_report(draft, citations, ledger, manifest)
        path = write_report(markdown, self._data_dir() / "reports", run_id)
        logger.info("render_done", report_path=str(path))
        return {"report_path": str(path)}
