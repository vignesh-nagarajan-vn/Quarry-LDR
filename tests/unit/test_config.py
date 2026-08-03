"""Config layering: defaults, default.yaml, user yaml, env vars."""

from __future__ import annotations

from pathlib import Path

import pytest

from quarry_ldr.config import QuarryConfig, deep_merge, load_config


def test_defaults_without_any_file() -> None:
    cfg = QuarryConfig(_env_file=None)
    assert cfg.run.max_iterations == 3
    assert cfg.models.plan == "claude-opus-5"
    assert cfg.gpu.vram_budget_mb == 6656
    assert cfg.chunk.target_tokens == 512
    assert cfg.anthropic_api_key is None


def test_default_yaml_matches_model_defaults() -> None:
    """config/default.yaml must never drift from the pydantic defaults."""
    from_defaults = QuarryConfig(_env_file=None)
    from_file = load_config()
    assert from_file.snapshot() == from_defaults.snapshot()


def test_user_yaml_overrides(tmp_path: Path) -> None:
    user = tmp_path / "user.yaml"
    user.write_text("run:\n  max_iterations: 5\nchunk:\n  target_tokens: 256\n", encoding="utf-8")
    cfg = load_config(user)
    assert cfg.run.max_iterations == 5
    assert cfg.chunk.target_tokens == 256
    # untouched keys keep defaults
    assert cfg.chunk.overlap_tokens == 64


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user = tmp_path / "user.yaml"
    user.write_text("run:\n  cost_cap_usd: 9.0\n", encoding="utf-8")
    monkeypatch.setenv("QUARRY_RUN__COST_CAP_USD", "1.25")
    cfg = load_config(user)
    assert cfg.run.cost_cap_usd == 1.25


def test_missing_user_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(bad)


def test_snapshot_excludes_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_key = "sk-ant-test-not-a-real-key"  # pragma: allowlist secret
    monkeypatch.setenv("ANTHROPIC_API_KEY", fake_key)
    cfg = QuarryConfig(_env_file=None)
    assert cfg.anthropic_api_key == fake_key
    snap = cfg.snapshot()
    assert "anthropic_api_key" not in snap
    assert "sk-ant" not in str(snap)


def test_deep_merge_nested_and_non_destructive() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10}, "e": 4}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3, "e": 4}
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}
