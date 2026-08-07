"""run_preflight: the checks quarry verify prints and research/resume gate on."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from quarry_ldr.config import QuarryConfig
from quarry_ldr.preflight import run_preflight

SERVER_BINARY_NAME = "llama-server.exe" if sys.platform == "win32" else "llama-server"


def _by_name(checks: list, name: str):
    return next(c for c in checks if c.name == name)


def test_local_mode_skips_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = QuarryConfig(_env_file=None)
    checks = run_preflight(cfg)
    key_check = _by_name(checks, "anthropic api key")
    assert key_check.status == "skip"
    assert "not needed for engine.mode=local" in key_check.detail


def test_premium_mode_requires_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = QuarryConfig(_env_file=None)
    cfg.engine.mode = "premium"
    checks = run_preflight(cfg)
    key_check = _by_name(checks, "anthropic api key")
    assert key_check.status == "missing"
    assert "ANTHROPIC_API_KEY" in key_check.detail


def test_missing_local_models_carries_download_remediation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An empty cwd has no models/ directory, so both the binary and the GGUFs
    # the pipeline needs are absent: this is the exact gap that used to crash
    # `quarry research` deep inside the arbiter instead of failing here.
    monkeypatch.chdir(tmp_path)
    cfg = QuarryConfig(_env_file=None)
    checks = run_preflight(cfg)
    models_check = _by_name(checks, "local models")
    assert models_check.status == "missing"
    assert "download_models.py" in models_check.detail


def test_finds_local_models_when_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    models_dir = tmp_path / "models"
    (models_dir / "llama.cpp").mkdir(parents=True)
    (models_dir / "llama.cpp" / SERVER_BINARY_NAME).write_bytes(b"")
    (models_dir / "gguf").mkdir(parents=True)
    (models_dir / "gguf" / "tiny.gguf").write_bytes(b"")
    (models_dir / "gguf" / "tiny-synth.gguf").write_bytes(b"")

    cfg = QuarryConfig(_env_file=None)
    cfg.run.models_dir = models_dir
    cfg.models.triage_gguf_file = "tiny.gguf"
    cfg.models.synth_gguf_file = "tiny-synth.gguf"

    checks = run_preflight(cfg)
    models_check = _by_name(checks, "local models")
    assert models_check.status == "ok"


def test_premium_mode_does_not_require_synth_gguf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    models_dir = tmp_path / "models"
    (models_dir / "llama.cpp").mkdir(parents=True)
    (models_dir / "llama.cpp" / SERVER_BINARY_NAME).write_bytes(b"")
    (models_dir / "gguf").mkdir(parents=True)
    (models_dir / "gguf" / "tiny.gguf").write_bytes(b"")

    cfg = QuarryConfig(_env_file=None)
    cfg.run.models_dir = models_dir
    cfg.models.triage_gguf_file = "tiny.gguf"
    cfg.engine.mode = "premium"

    checks = run_preflight(cfg)
    models_check = _by_name(checks, "local models")
    assert models_check.status == "ok"


def test_searxng_json_format_check_only_runs_when_settings_file_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = QuarryConfig(_env_file=None)
    checks = run_preflight(cfg)
    assert all(c.name != "searxng json format" for c in checks)

    searxng_dir = tmp_path / "docker" / "searxng"
    searxng_dir.mkdir(parents=True)
    (searxng_dir / "settings.yml").write_text(
        yaml.safe_dump({"search": {"formats": ["html"]}}), encoding="utf-8"
    )
    checks = run_preflight(cfg)
    format_check = _by_name(checks, "searxng json format")
    assert format_check.status == "missing"
