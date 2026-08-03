"""CLI surface: help works, commands exist, config errors are friendly."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from quarry_ldr.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("research", "resume", "inspect", "runs", "verify", "searxng"):
        assert command in result.output


def test_searxng_help() -> None:
    result = runner.invoke(app, ["searxng", "--help"])
    assert result.exit_code == 0
    for command in ("up", "down", "status"):
        assert command in result.output


def test_bad_config_path_is_friendly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["verify", "--config", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 2
    assert "config error" in result.output


def test_verify_reports_missing_pieces(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _: None)
    result = runner.invoke(app, ["verify"])
    # no docker and no API key in the test environment: preflight must fail
    # politely with remediation text, not crash.
    assert result.exit_code == 1
    assert "Docker" in result.output
    assert "ANTHROPIC_API_KEY" in result.output
