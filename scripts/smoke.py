"""End-to-end smoke test on real infrastructure with a hard $2.00 cost cap.

Runs the actual pipeline (real GPU, real SearXNG, and the real API when the
engine calls it) on a trivial topic, prints the ledger, checks the report
has resolvable citations, and exits nonzero if anything is off. This is the
one script the human runs at the end. ``--engine local`` is the $0
rehearsal (``make smoke-local``); ``--engine assisted``/``premium`` spend
real money under the cap.

Preflight runs first and prints exact remediation for anything missing
(exit 2): ANTHROPIC_API_KEY (skipped for the local engine, which makes zero
API calls), docker + a reachable SearXNG, a verified GPU, and the local
GGUFs the engine needs. Only once every check passes does it run
``Orchestrator(cfg).research(topic)``. A run that crosses the cost cap
(``CostCapExceeded``) prints the ledger persisted so far and exits 3; any
other failure to produce a report with resolvable citations exits 1.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import verify_gpu
from quarry_ldr.config import QuarryConfig, load_config
from quarry_ldr.gpu.local_llm import LlamaServerError, find_gguf, find_server_binary
from quarry_ldr.ingest.search import SearxClient
from quarry_ldr.ledger import CostCapExceeded, Ledger
from quarry_ldr.pipeline.run import Orchestrator
from quarry_ldr.state import RunStore

COST_CAP_USD = 2.00
MAX_ITERATIONS = 1
TOPIC = "what is a sand battery and how does one work"
ENGINE_MODES = ("local", "assisted", "premium")

DOCKER_REMEDIATION = (
    "install Docker Desktop (Windows/macOS) or Docker Engine (Linux), start it, "
    "then re-run this smoke test"
)
CITATION_MARKER = re.compile(r"\[(\d+)\]")
# Docker's Windows port proxy answers the first request after idle in ~3 s,
# so a 2 s probe misreads a healthy SearXNG as down.
_SEARXNG_PROBE_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str


def build_config(engine: str | None, max_cost: float | None) -> QuarryConfig:
    """Layered config with the smoke test's hard overrides applied."""
    cfg = load_config()
    if engine is not None:
        cfg.engine.mode = cast(Literal["local", "assisted", "premium"], engine)
    cfg.run.cost_cap_usd = max_cost if max_cost is not None else COST_CAP_USD
    cfg.run.max_iterations = MAX_ITERATIONS
    # Rehearsal breadth, matching the single-iteration depth cap: enough
    # sections to exercise cache write + reads + citations, not a full report.
    cfg.report.max_sections = 6
    return cfg


def preflight(cfg: QuarryConfig) -> list[PreflightCheck]:
    """Every precondition for a live run, each with exact remediation."""
    checks: list[PreflightCheck] = []

    key = cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if cfg.engine.mode == "local":
        # Local runs make zero API calls; a missing key is not a failure.
        checks.append(
            PreflightCheck(
                "ANTHROPIC_API_KEY",
                True,
                "found in environment" if key else "not needed for engine.mode=local",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "ANTHROPIC_API_KEY",
                bool(key),
                "found in environment"
                if key
                else "not set; fix: copy .env.example to .env and set "
                "ANTHROPIC_API_KEY=sk-ant-... (or export it in the shell)",
            )
        )

    if shutil.which("docker") is None:
        checks.append(
            PreflightCheck(
                "docker + searxng",
                False,
                f"docker is not on PATH; fix: {DOCKER_REMEDIATION}",
            )
        )
    else:
        try:
            healthy = asyncio.run(
                SearxClient(cfg.search.searxng_url, timeout_s=_SEARXNG_PROBE_TIMEOUT_S).health()
            )
        except Exception:
            healthy = False
        checks.append(
            PreflightCheck(
                "docker + searxng",
                healthy,
                f"reachable at {cfg.search.searxng_url}"
                if healthy
                else f"not reachable at {cfg.search.searxng_url}; fix: run "
                "`quarry searxng up` (or `make searxng`); requires Docker Desktop running",
            )
        )

    try:
        gpu_ok = verify_gpu.main() == 0
    except Exception as exc:
        gpu_ok = False
        print(f"scripts/verify_gpu.py raised unexpectedly: {exc}")
    checks.append(
        PreflightCheck(
            "gpu",
            gpu_ok,
            "verified" if gpu_ok else "failed; see the verify_gpu output above for the exact fix",
        )
    )

    models_dir = Path(cfg.run.models_dir)
    model_errors: list[str] = []
    try:
        find_server_binary(models_dir)
    except LlamaServerError as exc:
        model_errors.append(str(exc))
    needed = [cfg.models.triage_gguf_file]
    if cfg.engine.mode != "premium":
        # Local and assisted synthesis need the synth GGUF too.
        needed.append(cfg.models.synth_gguf_file)
    for gguf in needed:
        try:
            find_gguf(models_dir, gguf)
        except LlamaServerError as exc:
            model_errors.append(str(exc))
    checks.append(
        PreflightCheck(
            "local models",
            not model_errors,
            "llama-server binary and required GGUFs present"
            if not model_errors
            else "; ".join(model_errors),
        )
    )

    return checks


def validate_citations(report_text: str) -> list[str]:
    """Every [n] marker in the report body must resolve under References."""
    if "## References" not in report_text:
        return ["report has no '## References' section"]
    body, _, references = report_text.partition("## References")
    body_markers = {int(m) for m in CITATION_MARKER.findall(body)}
    ref_markers = {int(m) for m in CITATION_MARKER.findall(references)}
    missing = sorted(body_markers - ref_markers)
    if missing:
        return [f"citation marker(s) {missing} appear in the body but not in References"]
    return []


async def _print_ledger_for(cfg: QuarryConfig, run_id: str, label: str) -> None:
    async with RunStore(Path(cfg.run.data_dir) / "runs.db") as store:
        entries = await store.ledger_entries(run_id)
    ledger = Ledger.load(entries, cost_cap_usd=cfg.run.cost_cap_usd)
    print(f"\n{label}")
    print(ledger.to_markdown())


async def _run(cfg: QuarryConfig) -> tuple[int, Path | None]:
    orchestrator = Orchestrator(cfg)
    try:
        result = await orchestrator.research(TOPIC)
    except CostCapExceeded as exc:
        print(f"\nCOST CAP EXCEEDED: {exc}")
        async with RunStore(Path(cfg.run.data_dir) / "runs.db") as store:
            runs = await store.list_runs(limit=1)
        if runs:
            # The entry that crossed the cap raised before its stage's
            # checkpoint ran, so this is the ledger as of the last completed
            # stage, not the exact instant of the breach.
            await _print_ledger_for(cfg, runs[0].run_id, "ledger as of the last completed stage:")
        return 3, None

    print(
        f"\nrun {result.run_id} completed in {result.iterations} iteration(s): "
        f"{result.n_sources} sources, {result.n_chunks_indexed} chunks indexed, "
        f"{result.n_chunks_evidence} evidence chunks"
    )
    await _print_ledger_for(cfg, result.run_id, "cost ledger:")
    return 0, Path(result.report_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=ENGINE_MODES,
        default=None,
        help="Override engine.mode for this run (default: the configured mode).",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help=f"Override the hard cost cap in USD (default: {COST_CAP_USD:.2f}).",
    )
    args = parser.parse_args(argv)

    cfg = build_config(args.engine, args.max_cost)
    print(f"topic: {TOPIC!r}   engine: {cfg.engine.mode}")
    print(f"cost cap: ${cfg.run.cost_cap_usd:.2f}   max iterations: {cfg.run.max_iterations}")

    print("\npreflight:")
    checks = preflight(cfg)
    for check in checks:
        mark = "ok" if check.ok else "MISSING"
        print(f"  [{mark:<7}] {check.name}: {check.detail}")
    failed = [c for c in checks if not c.ok]
    if failed:
        print(f"\n{len(failed)} preflight check(s) failed; fix them and re-run `make smoke`.")
        return 2
    print("  all preflight checks passed")

    started = time.perf_counter()
    exit_code, report_path = asyncio.run(_run(cfg))
    if exit_code != 0:
        return exit_code
    assert report_path is not None

    if not report_path.is_file():
        print(f"FAIL: report file does not exist: {report_path}")
        return 1
    problems = validate_citations(report_path.read_text(encoding="utf-8"))
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1

    print(f"\nreport: {report_path}")
    print(f"elapsed: {time.perf_counter() - started:.1f}s")
    print("PASS: smoke run completed with resolvable citations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
