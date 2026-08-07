"""Typer CLI: the only layer allowed to print.

Commands that depend on later milestones fail with a clear message instead of
a traceback until their milestone lands.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from rich.console import Console

from quarry_ldr import __version__
from quarry_ldr.config import QuarryConfig, load_config
from quarry_ldr.gpu.local_llm import LlamaServerError
from quarry_ldr.logging import setup_logging
from quarry_ldr.preflight import DOCKER_REMEDIATION, PreflightCheck, run_preflight

app = typer.Typer(
    name="quarry",
    help="Quarry-LDR: local deep research; your GPU compresses the web and, "
    "by default, writes and verifies the cited report itself.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
searxng_app = typer.Typer(help="Manage the local SearXNG Docker instance.", no_args_is_help=True)
app.add_typer(searxng_app, name="searxng")

console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"quarry-ldr {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = False,
) -> None:
    """Quarry-LDR command line."""


ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="User config yaml layered over config/default.yaml."),
]

_ENGINE_MODES = ("local", "assisted", "premium")


def _apply_engine(cfg: QuarryConfig, engine: str | None) -> None:
    if engine is None:
        return
    if engine not in _ENGINE_MODES:
        err_console.print(
            f"[red]invalid --engine:[/red] {engine} (choose from {', '.join(_ENGINE_MODES)})"
        )
        raise typer.Exit(code=2)
    cfg.engine.mode = cast(Literal["local", "assisted", "premium"], engine)


def _load(config: Path | None) -> QuarryConfig:
    try:
        return load_config(config)
    except (FileNotFoundError, ValueError) as exc:
        err_console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


_STATUS_MARKS = {
    "ok": "[green]ok[/green]",
    "missing": "[red]missing[/red]",
    "skip": "[cyan]skip[/cyan]",
}


def _render_checks(checks: list[PreflightCheck], target: Console) -> int:
    """Print each check and return how many are missing."""
    failures = 0
    for check in checks:
        target.print(f"  {_STATUS_MARKS[check.status]:<20} {check.name}: {check.detail}")
        if check.status == "missing":
            failures += 1
    return failures


def _require_preflight(cfg: QuarryConfig) -> None:
    """Exit cleanly if a setup gap would otherwise crash deep in the pipeline."""
    failures = _render_checks(run_preflight(cfg), err_console)
    if failures:
        err_console.print(
            f"[yellow]{failures} check(s) need attention; run `quarry verify` for the "
            "full report.[/yellow]"
        )
        raise typer.Exit(code=1)


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
    engine: Annotated[
        str | None,
        typer.Option("--engine", help="Override engine.mode: local, assisted, or premium."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the full research pipeline and write a cited markdown report."""
    cfg = _load(config)
    if max_cost is not None:
        cfg.run.cost_cap_usd = max_cost
    if max_iterations is not None:
        cfg.run.max_iterations = max_iterations
    _apply_engine(cfg, engine)
    _require_preflight(cfg)
    setup_logging(log_dir=cfg.run.data_dir / "logs", verbose=verbose)

    from quarry_ldr.pipeline.run import Orchestrator

    try:
        result = asyncio.run(Orchestrator(cfg).research(topic))
    except LlamaServerError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]run:[/green] {result.run_id}  [green]engine:[/green] {cfg.engine.mode}")
    console.print(f"[green]report:[/green] {result.report_path}")
    if result.pdf_path:
        console.print(f"[green]pdf:[/green] {result.pdf_path}")
    console.print(f"[green]cost:[/green] ${result.total_cost_usd:.4f}")
    console.print(
        f"[green]iterations:[/green] {result.iterations}  "
        f"[green]sources:[/green] {result.n_sources}  "
        f"[green]evidence:[/green] {result.n_chunks_evidence}"
    )


@app.command()
def resume(
    run_id: Annotated[str, typer.Argument(help="Run id printed when the run started.")],
    config: ConfigOpt = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Resume an interrupted run from its last completed stage."""
    cfg = _load(config)
    _require_preflight(cfg)
    setup_logging(log_dir=cfg.run.data_dir / "logs", run_id=run_id, verbose=verbose)

    from quarry_ldr.pipeline.run import Orchestrator

    try:
        result = asyncio.run(Orchestrator(cfg).resume(run_id))
    except LlamaServerError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
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
    console.print(f"[bold]quarry preflight[/bold] (engine.mode={cfg.engine.mode})")
    failures = _render_checks(run_preflight(cfg), console)
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
