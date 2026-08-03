"""VRAM arbiter: the single owner of GPU model residency.

You cannot hold the embedder, the reranker, and a 4B LLM in ~7 GB at once.
The arbiter enforces a hard budget, evicts by LRU, serializes access with an
async lock, and logs measured VRAM before and after every load/evict so that
declared footprints get corrected against reality.

``async with arbiter.acquire("reranker") as model:`` is the ONLY way any code
touches a GPU model. The lock is held for the whole context, which also means
pipeline stages must batch their GPU work (embed everything, then swap once to
the reranker, then once to the local LLM) instead of interleaving. Never nest
``acquire`` calls: the lock is not reentrant and nesting deadlocks by design,
because nesting would mean two models interleaving.

Budget invariants (proved by tests against a fake backend):
  * the sum of resident footprints never exceeds ``budget_mb``, including
    at the moment a loader runs (eviction happens before loading);
  * a model whose footprint alone exceeds the budget raises
    :class:`BudgetExceededError` instead of thrashing;
  * measured footprints replace declared ones, and a post-load measurement
    that pushes the total over budget evicts LRU residents until it fits.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from quarry_ldr.logging import get_logger


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
    """Real backend over torch.cuda. Imported lazily; GPU-marked tests only."""

    def mem_info(self) -> tuple[int, int]:
        import torch

        free, total = torch.cuda.mem_get_info()
        return int(free), int(total)

    def synchronize(self) -> None:
        import torch

        torch.cuda.synchronize()

    def empty_cache(self) -> None:
        import torch

        torch.cuda.empty_cache()


@dataclass
class ModelSpec:
    """Registration record for a GPU model.

    ``loader`` returns the loaded model object; ``unloader`` releases it.
    ``footprint_mb`` is the declared VRAM cost used for budget math until a
    measured value replaces it.
    """

    name: str
    footprint_mb: int
    loader: Callable[[], Any]
    unloader: Callable[[Any], None]


@dataclass
class _Resident:
    spec: ModelSpec
    model: Any
    measured_mb: int | None = None

    @property
    def footprint_mb(self) -> int:
        return self.measured_mb if self.measured_mb is not None else self.spec.footprint_mb


class BudgetExceededError(Exception):
    """A single model's footprint exceeds the total budget: it can never load."""


class UnknownModelError(Exception):
    """acquire() was called for a name that was never registered."""


class VramArbiter:
    """Hard-budget LRU arbiter for GPU model residency."""

    def __init__(self, budget_mb: int, backend: GpuBackend | None = None) -> None:
        self.budget_mb = budget_mb
        self.backend = backend
        self._specs: dict[str, ModelSpec] = {}
        # dict insertion order doubles as LRU order: first key is coldest.
        self._resident: dict[str, _Resident] = {}
        self._lock = asyncio.Lock()
        self._log = get_logger(component="arbiter")

    def register(self, spec: ModelSpec) -> None:
        """Register a model. Re-registering an unloaded name replaces the spec."""
        if spec.name in self._resident:
            raise ValueError(f"cannot re-register {spec.name!r} while it is resident")
        if spec.footprint_mb > self.budget_mb:
            raise BudgetExceededError(
                f"{spec.name!r} declares {spec.footprint_mb} MB, over the whole "
                f"budget of {self.budget_mb} MB; it can never load"
            )
        self._specs[spec.name] = spec

    @asynccontextmanager
    async def acquire(self, name: str) -> AsyncIterator[Any]:
        """Yield the loaded model, loading and evicting as needed under the lock."""
        async with self._lock:
            spec = self._specs.get(name)
            if spec is None:
                known = ", ".join(sorted(self._specs)) or "(none registered)"
                raise UnknownModelError(f"unknown model {name!r}; registered: {known}")
            resident = self._resident.get(name)
            if resident is None:
                resident = self._load(spec)
            # Refresh LRU position: re-insert at the hot end.
            self._resident.pop(name)
            self._resident[name] = resident
            yield resident.model

    def _load(self, spec: ModelSpec) -> _Resident:
        self._evict_until_fits(spec.footprint_mb)
        used_before = self._used_mb()
        model = spec.loader()
        resident = _Resident(spec=spec, model=model)
        self._resident[spec.name] = resident
        used_after = self._used_mb()
        if used_before is not None and used_after is not None:
            measured = max(0, used_after - used_before)
            resident.measured_mb = measured
            if measured != spec.footprint_mb:
                self._log.info(
                    "footprint_corrected",
                    model=spec.name,
                    declared_mb=spec.footprint_mb,
                    measured_mb=measured,
                )
            if measured > self.budget_mb:
                # This model can never fit; undo the load and refuse.
                self._unload(spec.name)
                raise BudgetExceededError(
                    f"{spec.name!r} measured {measured} MB, over the whole "
                    f"budget of {self.budget_mb} MB"
                )
            # Measurement may reveal the total is now over budget: shed LRU
            # residents (the just-loaded model is hottest, evicted last).
            self._evict_until_fits(0, keep=spec.name)
        self._log.info(
            "model_loaded",
            model=spec.name,
            declared_mb=spec.footprint_mb,
            measured_mb=resident.measured_mb,
            resident=list(self._resident),
            resident_mb=self.resident_footprint_mb(),
            used_before_mb=used_before,
            used_after_mb=used_after,
        )
        return resident

    def _evict_until_fits(self, incoming_mb: int, keep: str | None = None) -> None:
        if incoming_mb > self.budget_mb:
            raise BudgetExceededError(
                f"a load of {incoming_mb} MB exceeds the whole budget of {self.budget_mb} MB"
            )
        while self.resident_footprint_mb() + incoming_mb > self.budget_mb:
            lru_name = next(
                (n for n in self._resident if n != keep),
                None,
            )
            if lru_name is None:  # pragma: no cover - keep-only residency cannot overflow
                raise BudgetExceededError(
                    f"cannot fit {incoming_mb} MB; nothing evictable within "
                    f"{self.budget_mb} MB budget"
                )
            self._unload(lru_name)

    def _unload(self, name: str) -> None:
        resident = self._resident.pop(name)
        used_before = self._used_mb()
        resident.spec.unloader(resident.model)
        if self.backend is not None:
            self.backend.empty_cache()
        used_after = self._used_mb()
        self._log.info(
            "model_evicted",
            model=name,
            footprint_mb=resident.footprint_mb,
            resident=list(self._resident),
            used_before_mb=used_before,
            used_after_mb=used_after,
        )

    def _used_mb(self) -> int | None:
        if self.backend is None:
            return None
        self.backend.synchronize()
        free, total = self.backend.mem_info()
        return (total - free) // (1024 * 1024)

    def resident_models(self) -> list[str]:
        """Names currently resident, LRU-first."""
        return list(self._resident)

    def resident_footprint_mb(self) -> int:
        """Sum of footprints of resident models (measured where available)."""
        return sum(resident.footprint_mb for resident in self._resident.values())

    def update_footprint(self, name: str, measured_mb: int) -> None:
        """Replace a declared footprint with a measured one."""
        spec = self._specs.get(name)
        if spec is None:
            raise UnknownModelError(f"unknown model {name!r}")
        spec.footprint_mb = measured_mb
        resident = self._resident.get(name)
        if resident is not None:
            resident.measured_mb = measured_mb

    async def evict_all(self) -> None:
        """Unload everything (end of run, or before handing VRAM to llama-server)."""
        async with self._lock:
            for name in list(self._resident):
                self._unload(name)
