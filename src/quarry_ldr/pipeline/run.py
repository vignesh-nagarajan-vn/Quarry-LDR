"""The orchestrator: a plain async state machine over the SQLite run store.

Every stage transition is a row; ``resume`` picks up from the last completed
stage with zero recomputation; a crash mid-fetch loses nothing because fetched
bodies live in the disk cache keyed by URL.

GPU stages are batched by model, never interleaved: all embedding, then one
swap to the reranker, then one swap to the local LLM.

Implemented incrementally: single pass in M8, iteration loop in M9.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from quarry_ldr.config import QuarryConfig


class RunResult(BaseModel):
    run_id: str
    report_path: str
    total_cost_usd: float
    iterations: int
    n_sources: int
    n_chunks_indexed: int
    n_chunks_evidence: int


class Orchestrator:
    """Owns component construction, stage sequencing, checkpointing, and the loop.

    Components (store, fetcher, embedder, provider, ...) are constructed
    lazily and are injectable for tests.
    """

    def __init__(self, cfg: QuarryConfig, **overrides: Any) -> None:
        self.cfg = cfg
        self._overrides = overrides

    async def research(self, topic: str) -> RunResult:
        """Create a run and execute PLAN through RENDER, looping per config."""
        raise NotImplementedError

    async def resume(self, run_id: str) -> RunResult:
        """Continue an interrupted run from its last completed stage."""
        raise NotImplementedError

    async def inspect(self, run_id: str) -> dict[str, Any]:
        """Stage-by-stage state as JSON-safe data for `quarry inspect`."""
        raise NotImplementedError
