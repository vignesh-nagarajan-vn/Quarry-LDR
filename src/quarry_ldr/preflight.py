"""Preflight checks shared by ``quarry verify`` and the run-starting commands.

``research``/``resume`` call :func:`run_preflight` before touching the
pipeline so a missing model, absent Docker daemon, or unset API key fails
with the same remediation text ``quarry verify`` prints, instead of
surfacing three stack frames deep inside the orchestrator.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.local_llm import LlamaServerError, find_gguf, find_server_binary

DOCKER_REMEDIATION = (
    "Docker is not available. Install Docker Desktop (Windows/macOS) or Docker "
    "Engine (Linux), start it, then re-run this command. SearXNG is only needed "
    "for live research runs; tests and fixture runs never touch it."
)


def _repo_root() -> Path:
    """Repo root that ships ``docker/`` and ``config/``.

    Anchored on this file's location, not the process cwd, so the SearXNG
    config checks validate the compose file ``quarry searxng up`` would
    actually launch. ``cli._compose`` and ``config.default_config_path``
    resolve the same way (``parents[2]``); preflight used a cwd-relative path
    and so reported the bundled config missing whenever ``quarry`` ran from
    anywhere but the repo root.
    """
    return Path(__file__).resolve().parents[2]


@dataclass
class PreflightCheck:
    """One preflight line: a name, a status, and the remediation detail."""

    name: str
    status: Literal["ok", "missing", "skip"]
    detail: str


def run_preflight(cfg: QuarryConfig) -> list[PreflightCheck]:
    """Report which runtime pieces are present, in the order verify prints them."""
    checks: list[PreflightCheck] = []

    key = cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if cfg.engine.mode == "local":
        # Local runs make zero API calls; a missing key is not a failure.
        detail = "found in environment" if key else "not needed for engine.mode=local"
        checks.append(PreflightCheck("anthropic api key", "skip", detail))
    else:
        checks.append(
            PreflightCheck(
                "anthropic api key",
                "ok" if key else "missing",
                "found in environment"
                if key
                else "set ANTHROPIC_API_KEY in .env (copy .env.example)",
            )
        )

    docker = shutil.which("docker")
    checks.append(
        PreflightCheck("docker", "ok" if docker else "missing", docker or DOCKER_REMEDIATION)
    )

    compose_path = _repo_root() / "docker" / "compose.yaml"
    compose_present = compose_path.is_file()
    checks.append(
        PreflightCheck(
            "searxng config",
            "ok" if compose_present else "missing",
            f"{compose_path} present" if compose_present else f"expected at {compose_path}",
        )
    )

    settings_path = _repo_root() / "docker" / "searxng" / "settings.yml"
    if settings_path.is_file():
        import yaml

        settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        formats = (settings.get("search") or {}).get("formats") or []
        json_enabled = "json" in formats
        checks.append(
            PreflightCheck(
                "searxng json format",
                "ok" if json_enabled else "missing",
                f"'json' enabled under search.formats in {settings_path}"
                if json_enabled
                else f"add 'json' under search.formats in {settings_path}",
            )
        )

    try:
        find_server_binary(cfg.run.models_dir)
        find_gguf(cfg.run.models_dir, cfg.models.triage_gguf_file)
        if cfg.engine.mode != "premium":
            # Local and assisted synthesis need the synth GGUF too.
            find_gguf(cfg.run.models_dir, cfg.models.synth_gguf_file)
        checks.append(PreflightCheck("local models", "ok", f"found under {cfg.run.models_dir}"))
    except LlamaServerError as exc:
        checks.append(PreflightCheck("local models", "missing", exc.remediation or str(exc)))

    try:
        import torch

        cuda = torch.cuda.is_available()
        detail = (
            f"{torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}"
            if cuda
            else "torch installed but CUDA unavailable; run scripts/verify_gpu.py"
        )
        checks.append(PreflightCheck("gpu (torch+cuda)", "ok" if cuda else "missing", detail))
    except ImportError:
        checks.append(
            PreflightCheck(
                "gpu (torch+cuda)", "missing", "install the GPU extra: uv sync --extra gpu"
            )
        )

    return checks
