"""Typer CLI: the only layer allowed to print.

Commands that depend on later milestones fail with a clear message instead of
a traceback until their milestone lands.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from quarry_ldr.config import QuarryConfig, load_config
from quarry_ldr.logging import setup_logging

app = typer.Typer(
    name="quarry",
    help="Quarry-LDR: local-GPU-compressed, Claude-synthesized deep research.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
searxng_app = typer.Typer(help="Manage the local SearXNG Docker instance.", no_args_is_help=True)
app.add_typer(searxng_app, name="searxng")

console = Console()
err_console = Console(stderr=True)

DOCKER_REMEDIATION = (
    "Docker is not available. Install Docker Desktop (Windows/macOS) or Docker "
    "Engine (Linux), start it, then re-run this command. SearXNG is only needed "
    "for live research runs; tests and fixture runs never touch it."
)

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="User config yaml layered over config/default.yaml."),
]


def _load(config: Path | None) -> QuarryConfig:
    try:
        return load_config(config)
    except (FileNotFoundError, ValueError) as exc:
        err_console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


@app.command()
def research(
    topic: Annotated[str, typer.Argument(help="Research topic for the report.")],
    config: ConfigOpt = None,
    max_cost: Annotated[
        float | None, typer.Option(help="Override run.cost_cap_usd for this run.")
    ] = None,
    max_iterations: Annotated[
        int | None, typer.Option(help="Override run.max_iterations for this run.")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the full research pipeline and write a cited markdown report."""
    cfg = _load(config)
    if max_cost is not None:
        cfg.run.cost_cap_usd = max_cost
    if max_iterations is not None:
        cfg.run.max_iterations = max_iterations
    setup_logging(log_dir=cfg.run.data_dir / "logs", verbose=verbose)

    from quarry_ldr.pipeline.run import Orchestrator

    result = asyncio.run(Orchestrator(cfg).research(topic))
    console.print(f"[green]report:[/green] {result.report_path}")
    console.print(f"[green]cost:[/green] ${result.total_cost_usd:.4f}")


@app.command()
def resume(
    run_id: Annotated[str, typer.Argument(help="Run id printed when the run started.")],
    config: ConfigOpt = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Resume an interrupted run from its last completed stage."""
    cfg = _load(config)
    setup_logging(log_dir=cfg.run.data_dir / "logs", run_id=run_id, verbose=verbose)

    from quarry_ldr.pipeline.run import Orchestrator

    result = asyncio.run(Orchestrator(cfg).resume(run_id))
    console.print(f"[green]report:[/green] {result.report_path}")


@app.command()
def inspect(
    run_id: Annotated[str, typer.Argument()],
    config: ConfigOpt = None,
) -> None:
    """Dump stage-by-stage run state as readable JSON."""
    cfg = _load(config)

    from quarry_ldr.pipeline.run import Orchestrator

    state = asyncio.run(Orchestrator(cfg).inspect(run_id))
    console.print_json(json.dumps(state, default=str))


@app.command()
def runs(config: ConfigOpt = None) -> None:
    """List known runs, newest first."""
    cfg = _load(config)

    from quarry_ldr.state import RunStore

    async def _list() -> None:
        async with RunStore(cfg.run.data_dir / "runs.db") as store:
            for record in await store.list_runs():
                console.print(
                    f"{record.run_id}  {record.status:<10} "
                    f"{record.created_at:%Y-%m-%d %H:%M}  {record.topic}"
                )

    asyncio.run(_list())


@app.command()
def verify(config: ConfigOpt = None) -> None:
    """Preflight: report which runtime pieces are present and how to fix gaps."""
    cfg = _load(config)
    failures = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        mark = "[green]ok[/green]" if ok else "[red]missing[/red]"
        console.print(f"  {mark:<20} {name}: {detail}")
        if not ok:
            failures += 1

    console.print("[bold]quarry preflight[/bold]")
    key = cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    check(
        "anthropic api key",
        bool(key),
        "found in environment" if key else "set ANTHROPIC_API_KEY in .env (copy .env.example)",
    )
    docker = shutil.which("docker")
    check("docker", docker is not None, docker or DOCKER_REMEDIATION)
    check(
        "searxng config",
        (Path("docker") / "compose.yaml").is_file(),
        "docker/compose.yaml present"
        if (Path("docker") / "compose.yaml").is_file()
        else "run from the repo root",
    )
    try:
        import torch

        cuda = torch.cuda.is_available()
        detail = (
            f"{torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}"
            if cuda
            else "torch installed but CUDA unavailable; run scripts/verify_gpu.py"
        )
        check("gpu (torch+cuda)", cuda, detail)
    except ImportError:
        check("gpu (torch+cuda)", False, "install the GPU extra: uv sync --extra gpu")

    if failures:
        console.print(f"[yellow]{failures} check(s) need attention. Live runs may fail.[/yellow]")
        raise typer.Exit(code=1)
    console.print("[green]all checks passed[/green]")


def _compose(args: list[str]) -> int:
    if shutil.which("docker") is None:
        err_console.print(f"[red]{DOCKER_REMEDIATION}[/red]")
        return 1
    compose_file = Path(__file__).resolve().parents[2] / "docker" / "compose.yaml"
    cmd = ["docker", "compose", "-f", str(compose_file), *args]
    return subprocess.run(cmd, check=False).returncode


@searxng_app.command("up")
def searxng_up() -> None:
    """Start SearXNG (docker compose up -d)."""
    code = _compose(["up", "-d"])
    if code == 0:
        console.print("SearXNG starting at http://localhost:8888 (JSON API enabled)")
    raise typer.Exit(code=code)


@searxng_app.command("down")
def searxng_down() -> None:
    """Stop SearXNG."""
    raise typer.Exit(code=_compose(["down"]))


@searxng_app.command("status")
def searxng_status() -> None:
    """Show SearXNG container status."""
    raise typer.Exit(code=_compose(["ps"]))


if __name__ == "__main__":
    app()
