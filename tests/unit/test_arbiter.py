"""Budget violation is provably impossible; LRU, serialization, measurement.

The fake backend simulates a VRAM allocator: loaders allocate, unloaders free,
mem_info reports the truth. The invariant checks run inside the loaders, at
the exact moment real VRAM would be committed."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from quarry_ldr.gpu.arbiter import (
    BudgetExceededError,
    ModelSpec,
    UnknownModelError,
    VramArbiter,
)

MB = 1024 * 1024
TOTAL_MB = 8192


class FakeGpuBackend:
    """Simulated allocator with mem_info in bytes, like torch.cuda.mem_get_info."""

    def __init__(self, total_mb: int = TOTAL_MB, baseline_mb: int = 600) -> None:
        self.total_mb = total_mb
        self.used_mb = baseline_mb  # OS / display baseline
        self.empty_cache_calls = 0

    def mem_info(self) -> tuple[int, int]:
        return (self.total_mb - self.used_mb) * MB, self.total_mb * MB

    def synchronize(self) -> None:
        pass

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def alloc(self, mb: int) -> None:
        self.used_mb += mb

    def free(self, mb: int) -> None:
        self.used_mb -= mb


class FakeModel:
    def __init__(self, name: str, size_mb: int) -> None:
        self.name = name
        self.size_mb = size_mb


def make_arbiter(
    budget_mb: int = 6500,
    footprints: dict[str, int] | None = None,
    actual_sizes: dict[str, int] | None = None,
) -> tuple[VramArbiter, FakeGpuBackend, list[str]]:
    """Arbiter + fake backend + event log. Loaders assert the budget invariant
    at allocation time: declared residency (excluding the incoming model) plus
    the incoming footprint must fit, or real VRAM would have been exceeded."""
    backend = FakeGpuBackend()
    arbiter = VramArbiter(budget_mb=budget_mb, backend=backend)
    events: list[str] = []
    footprints = footprints or {"embedder": 3000, "reranker": 2500, "triage": 2000}
    actual_sizes = actual_sizes or dict(footprints)

    for name, declared in footprints.items():
        actual = actual_sizes[name]

        def loader(name: str = name, declared: int = declared, actual: int = actual) -> FakeModel:
            resident_declared = sum(footprints[n] for n in arbiter.resident_models() if n != name)
            assert resident_declared + declared <= arbiter.budget_mb, (
                f"budget violated while loading {name}: "
                f"{resident_declared} + {declared} > {arbiter.budget_mb}"
            )
            backend.alloc(actual)
            events.append(f"load:{name}")
            return FakeModel(name, actual)

        def unloader(model: FakeModel) -> None:
            backend.free(model.size_mb)
            events.append(f"unload:{model.name}")

        arbiter.register(
            ModelSpec(name=name, footprint_mb=declared, loader=loader, unloader=unloader)
        )
    return arbiter, backend, events


async def test_budget_violation_is_impossible() -> None:
    """Three models totalling 7500 MB cannot co-reside in a 6500 MB budget.
    The invariant is asserted inside every loader and after every acquire."""
    arbiter, backend, events = make_arbiter()
    for name in ("embedder", "reranker", "triage", "embedder", "reranker"):
        async with arbiter.acquire(name) as model:
            assert isinstance(model, FakeModel)
            assert arbiter.resident_footprint_mb() <= arbiter.budget_mb
            assert backend.used_mb <= 600 + arbiter.budget_mb
    # The sequence forced evictions: all three never co-resided.
    assert any(event.startswith("unload:") for event in events)


async def test_lru_eviction_order() -> None:
    arbiter, _, events = make_arbiter()
    async with arbiter.acquire("embedder"):
        pass
    async with arbiter.acquire("reranker"):
        pass
    # Touch embedder again: reranker becomes LRU.
    async with arbiter.acquire("embedder"):
        pass
    # triage (2000) needs room: 3000 + 2500 + 2000 > 6500, evict LRU = reranker.
    async with arbiter.acquire("triage"):
        pass
    assert "unload:reranker" in events
    assert "unload:embedder" not in events
    assert arbiter.resident_models() == ["embedder", "triage"]


async def test_model_larger_than_budget_rejected_at_register() -> None:
    arbiter = VramArbiter(budget_mb=1000)
    with pytest.raises(BudgetExceededError, match="never load"):
        arbiter.register(
            ModelSpec(name="huge", footprint_mb=2000, loader=lambda: None, unloader=lambda m: None)
        )


async def test_unknown_model_rejected() -> None:
    arbiter, _, _ = make_arbiter()
    with pytest.raises(UnknownModelError, match="registered"):
        async with arbiter.acquire("nonexistent"):
            pass


async def test_access_is_fully_serialized() -> None:
    arbiter, _, _ = make_arbiter()
    inside = 0
    peak = 0

    async def use(name: str) -> None:
        nonlocal inside, peak
        async with arbiter.acquire(name):
            inside += 1
            peak = max(peak, inside)
            await asyncio.sleep(0.01)
            inside -= 1

    await asyncio.gather(use("embedder"), use("reranker"), use("triage"), use("embedder"))
    assert peak == 1


async def test_measured_footprint_corrects_declared() -> None:
    """Declared 1400 but reality is 2800: the resident record must carry the
    measured number so later budget math is honest."""
    arbiter, _backend, _ = make_arbiter(
        budget_mb=6500,
        footprints={"embedder": 1400, "reranker": 2500},
        actual_sizes={"embedder": 2800, "reranker": 2500},
    )
    async with arbiter.acquire("embedder"):
        pass
    assert arbiter.resident_footprint_mb() == 2800


async def test_overbudget_measurement_sheds_lru() -> None:
    """A load that measures far above its declaration pushes the total over
    budget; the arbiter must shed LRU residents to restore the invariant."""
    arbiter, _backend, events = make_arbiter(
        budget_mb=6500,
        footprints={"embedder": 3000, "reranker": 1000},
        actual_sizes={"embedder": 3000, "reranker": 4000},
    )
    async with arbiter.acquire("embedder"):
        pass
    async with arbiter.acquire("reranker"):
        pass
    assert "unload:embedder" in events
    assert arbiter.resident_models() == ["reranker"]
    assert arbiter.resident_footprint_mb() <= 6500


async def test_reuse_does_not_reload() -> None:
    arbiter, _, events = make_arbiter()
    async with arbiter.acquire("embedder"):
        pass
    async with arbiter.acquire("embedder"):
        pass
    assert events.count("load:embedder") == 1


async def test_evict_all_frees_everything() -> None:
    arbiter, backend, _events = make_arbiter()
    async with arbiter.acquire("embedder"):
        pass
    async with arbiter.acquire("reranker"):
        pass
    await arbiter.evict_all()
    assert arbiter.resident_models() == []
    assert arbiter.resident_footprint_mb() == 0
    assert backend.used_mb == 600  # back to baseline
    assert backend.empty_cache_calls >= 2


async def test_update_footprint_applies_to_spec_and_resident() -> None:
    arbiter, _, _ = make_arbiter()
    async with arbiter.acquire("embedder"):
        pass
    arbiter.update_footprint("embedder", 3333)
    assert arbiter.resident_footprint_mb() == 3333
    with pytest.raises(UnknownModelError):
        arbiter.update_footprint("nope", 1)


async def test_works_without_backend() -> None:
    """CPU-only environments have no backend; declared numbers still govern."""
    arbiter = VramArbiter(budget_mb=5000, backend=None)
    loads: list[str] = []
    arbiter.register(
        ModelSpec(
            name="a", footprint_mb=3000, loader=lambda: loads.append("a"), unloader=lambda m: None
        )
    )
    arbiter.register(
        ModelSpec(
            name="b", footprint_mb=3000, loader=lambda: loads.append("b"), unloader=lambda m: None
        )
    )
    async with arbiter.acquire("a"):
        pass
    async with arbiter.acquire("b"):
        pass
    assert arbiter.resident_models() == ["b"]  # a evicted, declared math only


async def test_register_while_resident_rejected() -> None:
    arbiter, _, _ = make_arbiter()
    async with arbiter.acquire("embedder"):
        pass
    with pytest.raises(ValueError, match="resident"):
        arbiter.register(
            ModelSpec(
                name="embedder", footprint_mb=100, loader=lambda: None, unloader=lambda m: None
            )
        )


def test_fake_backend_satisfies_protocol_surface() -> None:
    from quarry_ldr.gpu.arbiter import TorchCudaBackend

    backend: Any = FakeGpuBackend()
    for attr in ("mem_info", "synchronize", "empty_cache"):
        assert callable(getattr(backend, attr))
        assert hasattr(TorchCudaBackend, attr)
