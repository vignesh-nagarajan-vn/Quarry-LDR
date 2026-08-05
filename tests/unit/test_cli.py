"""CLI surface: help works, commands exist, config errors are friendly."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from quarry_ldr.cli import app

runner = CliRunner()

SERVER_BINARY_NAME = "llama-server.exe" if sys.platform == "win32" else "llama-server"


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


def test_verify_reports_missing_pieces(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import shutil

    # chdir away from the repo so a developer's real .env cannot satisfy the
    # API key check; the CLI reading .env from cwd is correct product behavior.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    # The key check only applies to API-calling engines; pin premium.
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump({"engine": {"mode": "premium"}}), encoding="utf-8")
    result = runner.invoke(app, ["verify", "--config", str(config_path)])
    # no docker and no API key in the test environment: preflight must fail
    # politely with remediation text, not crash.
    assert result.exit_code == 1
    assert "Docker" in result.output
    assert "ANTHROPIC_API_KEY" in result.output


def test_verify_local_mode_skips_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Default engine is local: a missing key is a skip line, not a failure.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["verify"])
    assert "not needed for engine.mode=local" in result.output
    assert "set ANTHROPIC_API_KEY" not in result.output


def test_research_rejects_bad_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["research", "some topic", "--engine", "turbo"])
    assert result.exit_code == 2
    assert "invalid --engine" in result.output


def test_verify_reports_missing_local_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An empty cwd has no models/ directory under the default run.models_dir,
    # so both the server binary and the GGUF are absent.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["verify"])
    assert "local models" in result.output
    assert "download_models.py" in result.output


def test_verify_finds_local_models_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    models_dir = tmp_path / "models"
    (models_dir / "llama.cpp").mkdir(parents=True)
    (models_dir / "llama.cpp" / SERVER_BINARY_NAME).write_bytes(b"")
    (models_dir / "gguf").mkdir(parents=True)
    (models_dir / "gguf" / "tiny.gguf").write_bytes(b"")

    (models_dir / "gguf" / "tiny-synth.gguf").write_bytes(b"")

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run": {"models_dir": str(models_dir)},
                # The default (local) engine requires the synth GGUF too.
                "models": {"triage_gguf_file": "tiny.gguf", "synth_gguf_file": "tiny-synth.gguf"},
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["verify", "--config", str(config_path)])
    assert "local models" in result.output
    assert "download_models.py" not in result.output


def test_verify_searxng_json_format_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    searxng_dir = tmp_path / "docker" / "searxng"
    searxng_dir.mkdir(parents=True)
    (searxng_dir / "settings.yml").write_text(
        yaml.safe_dump({"search": {"formats": ["html", "json"]}}), encoding="utf-8"
    )

    result = runner.invoke(app, ["verify"])
    assert "searxng json format" in result.output
    assert "enabled" in result.output


def test_verify_searxng_json_format_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    searxng_dir = tmp_path / "docker" / "searxng"
    searxng_dir.mkdir(parents=True)
    (searxng_dir / "settings.yml").write_text(
        yaml.safe_dump({"search": {"formats": ["html"]}}), encoding="utf-8"
    )

    result = runner.invoke(app, ["verify"])
    assert "searxng json format" in result.output
    assert "add 'json'" in result.output


def test_verify_skips_searxng_json_check_when_settings_file_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["verify"])
    assert "searxng json format" not in result.output
