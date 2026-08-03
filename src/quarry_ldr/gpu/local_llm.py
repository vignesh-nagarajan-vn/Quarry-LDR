"""llama-server lifecycle plus an OpenAI-compatible client for local triage.

The server binary is downloaded by scripts/download_models.py into
``models/llama.cpp/``. The server process is registered with the arbiter as
"triage" so its VRAM (weights + KV cache) participates in budget math; its
loader starts the subprocess and waits for /health, its unloader terminates it.

Implemented in M6.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import httpx
from pydantic import BaseModel

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.arbiter import VramArbiter

T = TypeVar("T", bound=BaseModel)


class LlamaServerError(Exception):
    """Server failed to start, became unhealthy, or the binary is missing.

    Carries a ``remediation`` string with the exact fix, e.g. the
    download_models.py invocation or the port to free.
    """

    def __init__(self, message: str, remediation: str = "") -> None:
        self.remediation = remediation
        super().__init__(message if not remediation else f"{message}\nfix: {remediation}")


def find_server_binary(models_dir: Path) -> Path:
    """Locate llama-server(.exe) under models_dir, raising with remediation if absent."""
    raise NotImplementedError


class LlamaServer:
    """Manages the llama-server subprocess as an arbiter-registered model."""

    ARBITER_NAME = "triage"

    def __init__(self, cfg: QuarryConfig, arbiter: VramArbiter, models_dir: Path) -> None:
        self.cfg = cfg
        self.arbiter = arbiter
        self.models_dir = models_dir
        self.port = cfg.triage.port

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        """Spawn llama-server with full GPU offload and wait until /health is ok."""
        raise NotImplementedError

    async def stop(self) -> None:
        """Terminate the subprocess and wait for exit."""
        raise NotImplementedError

    async def is_healthy(self) -> bool:
        raise NotImplementedError


class LocalLLM:
    """Minimal client for llama-server's OpenAI-compatible /v1/chat/completions."""

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout_s = timeout_s
        self._client = client

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        json_schema: dict[str, object] | None = None,
    ) -> str:
        """Single-turn completion; json_schema enables grammar-constrained output."""
        raise NotImplementedError

    async def complete_typed(
        self,
        prompt: str,
        schema: type[T],
        max_tokens: int = 512,
        max_retries: int = 2,
    ) -> T:
        """Completion parsed into a pydantic model, retrying on malformed JSON."""
        raise NotImplementedError
