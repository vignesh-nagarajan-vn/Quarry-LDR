"""Measure real VRAM footprints and throughput for every registered model.

Loads each model alone under the arbiter, records torch.cuda.mem_get_info
before and after, runs a small workload for tokens/sec (the laptop thermal
story), and prints a table to paste into DECISIONS.md and config
gpu.footprints_mb.

Requires a real GPU: exits 1 with remediation (never a traceback) when torch
or CUDA is absent, same contract as scripts/verify_gpu.py. Everything else
here needs the actual hardware, so there are no unit tests for this module.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quarry_ldr.config import QuarryConfig, load_config
from quarry_ldr.gpu.arbiter import TorchCudaBackend, VramArbiter
from quarry_ldr.gpu.embedder import Embedder
from quarry_ldr.gpu.local_llm import LlamaServer, LlamaServerError, LocalLLM, synth_server_spec
from quarry_ldr.gpu.reranker import Reranker
from quarry_ldr.pipeline.triage import TRIAGE_PROMPT, TriageVerdict

# 64 short texts / 64 pairs, per the spec: enough to see steady-state
# throughput without turning this into a real benchmark suite.
_WORKLOAD_N = 64
_QUESTION = "What temperature does the sand bed reach during charging?"
_PASSAGE = (
    "Resistive heaters raise the sand bed to 600 degrees Celsius during "
    "off-peak hours, storing heat for later district discharge."
)


@dataclass(frozen=True)
class BenchResult:
    model: str
    declared_mb: int
    measured_mb: int | None
    throughput: str
    note: str = ""


def _used_mb() -> int:
    """Used VRAM in MB on the current device, synchronized first."""
    import torch

    torch.cuda.synchronize()
    free_b, total_b = torch.cuda.mem_get_info()
    return (total_b - free_b) // (1024 * 1024)


async def _bench_embedder(cfg: QuarryConfig, arbiter: VramArbiter) -> BenchResult:
    embedder = Embedder(cfg, arbiter)
    texts = [
        f"Sample passage {i} about sand-battery grid storage economics." for i in range(_WORKLOAD_N)
    ]

    used_before = _used_mb()
    async with arbiter.acquire(Embedder.ARBITER_NAME) as model:
        used_after_load = _used_mb()
        start = time.perf_counter()
        await asyncio.to_thread(
            model.encode,
            texts,
            batch_size=embedder.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        elapsed = time.perf_counter() - start

    declared_mb = cfg.gpu.footprints_mb.get("embedder", 2186)
    measured_mb = max(0, used_after_load - used_before)
    throughput = f"{_WORKLOAD_N / elapsed:.1f} texts/s" if elapsed > 0 else "n/a"
    return BenchResult("embedder", declared_mb, measured_mb, throughput)


async def _bench_reranker(cfg: QuarryConfig, arbiter: VramArbiter) -> BenchResult:
    reranker = Reranker(cfg, arbiter)
    pairs = [(_QUESTION, f"{_PASSAGE} (variant {i})") for i in range(_WORKLOAD_N)]

    used_before = _used_mb()
    async with arbiter.acquire(Reranker.ARBITER_NAME) as model:
        used_after_load = _used_mb()
        start = time.perf_counter()
        await asyncio.to_thread(model.predict, pairs, batch_size=reranker.batch_size)
        elapsed = time.perf_counter() - start

    declared_mb = cfg.gpu.footprints_mb.get("reranker", 2128)
    measured_mb = max(0, used_after_load - used_before)
    throughput = f"{_WORKLOAD_N / elapsed:.1f} pairs/s" if elapsed > 0 else "n/a"
    return BenchResult("reranker", declared_mb, measured_mb, throughput)


async def _bench_llama_server(
    cfg: QuarryConfig, arbiter: VramArbiter, server: LlamaServer, label: str
) -> BenchResult | None:
    declared_mb = cfg.gpu.footprints_mb.get(
        server.spec.footprint_key, server.spec.default_footprint_mb
    )
    try:
        await server.start()
    except LlamaServerError as exc:
        print(f"SKIP: {label}: {exc}")
        return None

    local_llm = LocalLLM(server.base_url)
    if server.spec.footprint_key == "triage":
        prompt = TRIAGE_PROMPT.format(question=_QUESTION, passage=_PASSAGE)
        start = time.perf_counter()
        await local_llm.complete_typed(prompt, TriageVerdict, max_retries=cfg.triage.max_retries)
        elapsed = time.perf_counter() - start
        throughput = f"1 verdict in {elapsed:.2f}s"
    else:
        # Section-shaped workload: sustained generation, reported as tok/s.
        start = time.perf_counter()
        _, usage, _ = await local_llm.complete_with_usage(
            "Write a 150-word overview of sand-battery grid storage economics.",
            max_tokens=256,
        )
        elapsed = time.perf_counter() - start
        tok_s = usage.output_tokens / elapsed if elapsed > 0 else 0.0
        throughput = f"{usage.output_tokens} tokens in {elapsed:.2f}s ({tok_s:.1f} tok/s)"

    return BenchResult(
        label,
        declared_mb,
        None,
        throughput,
        note="WDDM hides a child process's VRAM from the parent; measured MB "
        "is not observable here, only the declared footprint.",
    )


async def _run_all(cfg: QuarryConfig, arbiter: VramArbiter) -> list[BenchResult]:
    results = [await _bench_embedder(cfg, arbiter), await _bench_reranker(cfg, arbiter)]
    models_dir = Path(cfg.run.models_dir)
    triage_server = LlamaServer(cfg, arbiter, models_dir)
    synth_server = LlamaServer(cfg, arbiter, models_dir, spec=synth_server_spec(cfg))
    try:
        for server, label in (
            (triage_server, "triage (llama-server)"),
            (synth_server, "synth (llama-server)"),
        ):
            result = await _bench_llama_server(cfg, arbiter, server, label)
            if result is not None:
                results.append(result)
    finally:
        await arbiter.evict_all()
    return results


def _print_table(results: list[BenchResult]) -> None:
    print()
    print("| Model | Declared MB | Measured MB | Throughput |")
    print("| --- | --- | --- | --- |")
    for r in results:
        measured = str(r.measured_mb) if r.measured_mb is not None else "n/a"
        print(f"| {r.model} | {r.declared_mb} | {measured} | {r.throughput} |")
    for r in results:
        if r.note:
            print(f"\nnote ({r.model}): {r.note}")
    print(
        "\nUpdate config/default.yaml gpu.footprints_mb with these measured "
        "values where a real number was observed."
    )


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed.")
        print("fix:  uv sync --extra gpu")
        return 1
    if not torch.cuda.is_available():
        print("FAIL: torch imports but CUDA is not available.")
        print("fix:  see scripts/verify_gpu.py for the full diagnostic.")
        return 1

    cfg = load_config()
    arbiter = VramArbiter(cfg.gpu.vram_budget_mb, TorchCudaBackend())

    results = asyncio.run(_run_all(cfg, arbiter))
    _print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
