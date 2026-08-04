"""Shared test fixtures.

The default test selection runs on CPU with no network (pytest-socket blocks
sockets except loopback) and no API key. GPU and live tests are opt-in via
markers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from quarry_ldr.config import QuarryConfig

# With a warm model cache, huggingface_hub still makes a per-file etag check
# over the network at load time, which pytest-socket rightly blocks. Offline
# mode makes gpu-marked tests deterministic on any machine whose cache already
# holds the models (first fetched outside pytest, e.g. by scripts/bench_vram.py).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture()
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(autouse=True)
def _clean_quarry_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never inherit QUARRY_* overrides or a real API key from the host."""
    import os

    for name in list(os.environ):
        if name.startswith("QUARRY_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture()
def cfg(tmp_path: Path) -> QuarryConfig:
    """Default config pointed at a temp data dir; no env, no yaml files."""
    config = QuarryConfig(_env_file=None)
    config.run.data_dir = tmp_path / "data"
    config.fetch.cache_dir = tmp_path / "data" / "cache" / "fetch"
    return config
