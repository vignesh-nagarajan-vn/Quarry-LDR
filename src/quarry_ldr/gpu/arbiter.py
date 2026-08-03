"""VRAM arbiter: the single owner of GPU model residency.

You cannot hold the embedder, the reranker, and a 4B LLM in ~7 GB at once.
The arbiter enforces a hard budget, evicts by LRU, serializes access with an
async lock, and logs measured VRAM before and after every load/evict so that
declared footprints get corrected against reality.

``async with arbiter.acquire("reranker") as model:`` is the ONLY way any code
touches a GPU model. The lock is held for the whole context, which also means
pipeline stages must batch their GPU work (embed everything, then swap once to
the reranker, then once to the local LLM) instead of interleaving.

Implemented in M2.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol


class GpuBackend(Protocol):
    """Minimal surface the arbiter needs from the GPU runtime."""

    def mem_info(self) -> tuple[int, int]:
        """Return (free_bytes, total_bytes) on the device."""
        ...

    def synchronize(self) -> None:
        """Block until pending device work completes (before measuring)."""
        ...

    def empty_cache(self) -> None:
        """Release cached allocator blocks after an eviction."""
        ...


class TorchCudaBackend:
    """Real backend over torch.cuda. Imports torch lazily; GPU-marked tests only."""

    def mem_info(self) -> tuple[int, int]:
        raise NotImplementedError

    def synchronize(self) -> None:
        raise NotImplementedError

    def empty_cache(self) -> None:
        raise NotImplementedError


@dataclass
class ModelSpec:
    """Registration record for a GPU model.

    ``loader`` returns the loaded model object; ``unloader`` releases it.
    ``footprint_mb`` is the declared VRAM cost used for budget math until a
    measured value replaces it via :meth:`VramArbiter.update_footprint`.
    """

    name: str
    footprint_mb: int
    loader: Callable[[], Any]
    unloader: Callable[[Any], None]


@dataclass
class _Resident:
    spec: ModelSpec
    model: Any
    loaded_at_seq: int
    last_used_seq: int = 0
    measured_mb: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BudgetExceededError(Exception):
    """A single model's footprint exceeds the total budget: it can never load."""


class UnknownModelError(Exception):
    """acquire() was called for a name that was never registered."""


class VramArbiter:
    """Hard-budget LRU arbiter for GPU model residency.

    Invariants (tested):
      * The sum of resident footprints never exceeds ``budget_mb``, at any
        point in time, including transiently during load.
      * Access is fully serialized: while one ``acquire`` context is open no
        other coroutine can load, evict, or use a model.
      * A model whose footprint alone exceeds the budget raises
        :class:`BudgetExceededError` instead of thrashing.
    """

    def __init__(self, budget_mb: int, backend: GpuBackend | None = None) -> None:
        self.budget_mb = budget_mb
        self.backend = backend
        self._specs: dict[str, ModelSpec] = {}
        self._resident: dict[str, _Resident] = {}

    def register(self, spec: ModelSpec) -> None:
        """Register a model. Re-registering an unloaded name replaces the spec."""
        raise NotImplementedError

    @asynccontextmanager
    async def acquire(self, name: str) -> AsyncIterator[Any]:
        """Yield the loaded model, loading and evicting as needed under the lock."""
        raise NotImplementedError
        yield  # pragma: no cover

    def resident_models(self) -> list[str]:
        """Names currently resident, LRU-first."""
        raise NotImplementedError

    def resident_footprint_mb(self) -> int:
        """Sum of footprints of resident models (measured where available)."""
        raise NotImplementedError

    def update_footprint(self, name: str, measured_mb: int) -> None:
        """Replace a declared footprint with a measured one."""
        raise NotImplementedError

    async def evict_all(self) -> None:
        """Unload everything (end of run, or before handing VRAM to llama-server)."""
        raise NotImplementedError
