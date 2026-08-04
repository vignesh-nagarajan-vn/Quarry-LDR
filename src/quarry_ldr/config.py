"""Layered configuration: pydantic model defaults < config/default.yaml < user yaml < env.

Every default in ``config/default.yaml`` is duplicated here so a missing file
never breaks a run. Environment variables use the ``QUARRY_`` prefix with
``__`` as the nesting delimiter, e.g. ``QUARRY_RUN__COST_CAP_USD=2.5``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class RunSettings(BaseModel):
    max_iterations: int = 3
    cost_cap_usd: float = 5.0
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")


class ModelSettings(BaseModel):
    plan: str = "claude-opus-5"
    gap: str = "claude-sonnet-5"
    synthesize: str = "claude-opus-5"
    extract_fallback: str = "claude-haiku-4-5-20251001"
    embedder: str = "BAAI/bge-m3"
    reranker: str = "BAAI/bge-reranker-v2-m3"
    triage_gguf_repo: str = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    triage_gguf_file: str = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"


class GpuSettings(BaseModel):
    vram_budget_mb: int = 6656
    footprints_mb: dict[str, int] = Field(
        default_factory=lambda: {"embedder": 2186, "reranker": 2128, "triage": 3600}
    )
    embed_batch_size: int = 32
    rerank_batch_size: int = 16


class SearchSettings(BaseModel):
    searxng_url: str = "http://localhost:8888"
    results_per_query: int = 10
    timeout_s: float = 15.0


class FetchSettings(BaseModel):
    user_agent: str = "QuarryLDR/0.1 (+https://github.com/vignesh-nagarajan-vn/Quarry-LDR)"
    per_domain_rps: float = 1.0
    max_concurrency: int = 8
    timeout_s: float = 20.0
    max_bytes: int = 5_000_000
    cache_dir: Path = Path("data/cache/fetch")


class ChunkSettings(BaseModel):
    target_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 48


class DedupSettings(BaseModel):
    cosine_threshold: float = 0.92
    simhash_hamming_max: int = 3


class RetrieveSettings(BaseModel):
    ann_top_k: int = 200
    rerank_top_k: int = 40


class TriageSettings(BaseModel):
    context_tokens: int = 8192
    port: int = 8555
    max_retries: int = 2
    confidence_floor: float = 0.3


class ApiSettings(BaseModel):
    max_retries: int = 5
    retry_base_s: float = 1.0
    cache_ttl: str = "1h"
    batch_poll_s: float = 30.0


class ReportSettings(BaseModel):
    min_sections: int = 4
    max_sections: int = 12
    corpus_budget_tokens: int = 60000


class QuarryConfig(BaseSettings):
    """Root configuration object passed through the whole pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="QUARRY_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    run: RunSettings = Field(default_factory=RunSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    gpu: GpuSettings = Field(default_factory=GpuSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    fetch: FetchSettings = Field(default_factory=FetchSettings)
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    dedup: DedupSettings = Field(default_factory=DedupSettings)
    retrieve: RetrieveSettings = Field(default_factory=RetrieveSettings)
    triage: TriageSettings = Field(default_factory=TriageSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    report: ReportSettings = Field(default_factory=ReportSettings)

    # Sourced from the conventional env var, never from yaml. repr=False and
    # exclusion from snapshots keep it out of logs and the run store.
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY", repr=False)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # First source wins: env > .env > yaml (passed via init) > model defaults.
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe dump for the run store, with the API key excluded."""
        return self.model_dump(mode="json", exclude={"anthropic_api_key"})


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a mapping, got {type(data).__name__}")
    return data


def default_config_path() -> Path | None:
    """Locate config/default.yaml relative to the repo root, if present."""
    candidate = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    return candidate if candidate.is_file() else None


def load_config(user_config: Path | None = None) -> QuarryConfig:
    """Build the layered configuration.

    Precedence, lowest to highest: pydantic defaults, config/default.yaml,
    the ``user_config`` file (or ``QUARRY_CONFIG`` env var), ``QUARRY_*``
    environment variables.
    """
    merged: dict[str, Any] = {}
    default_path = default_config_path()
    if default_path is not None:
        merged = deep_merge(merged, _load_yaml(default_path))

    if user_config is None:
        env_path = os.environ.get("QUARRY_CONFIG")
        if env_path:
            user_config = Path(env_path)
    if user_config is not None:
        if not user_config.is_file():
            raise FileNotFoundError(f"config file not found: {user_config}")
        merged = deep_merge(merged, _load_yaml(user_config))

    return QuarryConfig(**merged)
