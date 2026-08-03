"""llama-server lifecycle plus an OpenAI-compatible client for local triage.

The server binary and GGUF are downloaded by scripts/download_models.py into
``models/``. The server process is registered with the arbiter as "triage" so
its VRAM (weights plus KV cache) participates in budget math: the loader
spawns the subprocess with full GPU offload and blocks until /health is ok,
the unloader terminates it. Acquiring "triage" therefore evicts the embedder
and reranker first when the budget requires it, which is exactly the batched
stage order the pipeline uses.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.arbiter import ModelSpec, VramArbiter
from quarry_ldr.logging import get_logger

logger = get_logger(component="local_llm")

T = TypeVar("T", bound=BaseModel)

DOWNLOAD_REMEDIATION = "run: uv run python scripts/download_models.py"
_HEALTH_TIMEOUT_S = 180.0
_HEALTH_POLL_S = 1.0


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
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    candidates = [models_dir / "llama.cpp" / name, *sorted(models_dir.glob(f"llama.cpp/**/{name}"))]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise LlamaServerError(
        f"llama-server binary not found under {models_dir / 'llama.cpp'}",
        DOWNLOAD_REMEDIATION,
    )


def find_gguf(models_dir: Path, filename: str) -> Path:
    """Locate the triage GGUF, raising with remediation if absent."""
    candidates = [models_dir / "gguf" / filename, *sorted(models_dir.glob(f"gguf/**/{filename}"))]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise LlamaServerError(
        f"GGUF {filename!r} not found under {models_dir / 'gguf'}", DOWNLOAD_REMEDIATION
    )


@dataclass
class _ServerHandle:
    process: subprocess.Popen[bytes]
    port: int


class LlamaServer:
    """Manages the llama-server subprocess as an arbiter-registered model."""

    ARBITER_NAME = "triage"

    def __init__(self, cfg: QuarryConfig, arbiter: VramArbiter, models_dir: Path) -> None:
        self.cfg = cfg
        self.arbiter = arbiter
        self.models_dir = models_dir
        self.port = cfg.triage.port
        arbiter.register(
            ModelSpec(
                name=self.ARBITER_NAME,
                footprint_mb=cfg.gpu.footprints_mb.get("triage", 3600),
                loader=self._spawn,
                unloader=self._terminate,
            )
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _command(self, binary: Path, gguf: Path) -> list[str]:
        return [
            str(binary),
            "-m",
            str(gguf),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-ngl",
            "99",  # full offload; partial offload is catastrophic on 8 GB cards
            "-c",
            str(self.cfg.triage.context_tokens),
            "--jinja",  # use the GGUF's embedded chat template
        ]

    def _spawn(self) -> _ServerHandle:
        """Arbiter loader: spawn the subprocess and block until /health is ok."""
        # Absolute paths: the subprocess runs with cwd at the binary's folder
        # (so its DLLs resolve), which would break a relative -m path.
        binary = find_server_binary(self.models_dir).resolve()
        gguf = find_gguf(self.models_dir, self.cfg.models.triage_gguf_file).resolve()
        command = self._command(binary, gguf)
        logger.info("llama_server_starting", port=self.port, gguf=gguf.name)
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(binary.parent),
            )
        except OSError as exc:
            raise LlamaServerError(
                f"could not spawn llama-server: {exc}", DOWNLOAD_REMEDIATION
            ) from exc

        deadline = time.monotonic() + _HEALTH_TIMEOUT_S
        with httpx.Client(timeout=2.0) as client:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise LlamaServerError(
                        f"llama-server exited with code {process.returncode} during startup",
                        f"port {self.port} may be in use (change triage.port), or the "
                        f"binary/GGUF may be broken; {DOWNLOAD_REMEDIATION}",
                    )
                try:
                    response = client.get(f"{self.base_url}/health")
                    if response.status_code == 200:
                        logger.info("llama_server_healthy", port=self.port)
                        return _ServerHandle(process=process, port=self.port)
                except httpx.HTTPError:
                    pass
                time.sleep(_HEALTH_POLL_S)
        process.terminate()
        raise LlamaServerError(
            f"llama-server did not become healthy within {_HEALTH_TIMEOUT_S:.0f}s",
            "slow disk or too-large context; lower triage.context_tokens",
        )

    def _terminate(self, handle: _ServerHandle) -> None:
        """Arbiter unloader: terminate the subprocess and wait for exit."""
        process = handle.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        logger.info("llama_server_stopped", port=handle.port)

    async def start(self) -> None:
        """Load (spawn) the server via the arbiter, evicting others as needed."""
        async with self.arbiter.acquire(self.ARBITER_NAME):
            pass

    async def stop(self) -> None:
        """Evict everything, terminating the server if it is resident."""
        await self.arbiter.evict_all()

    async def is_healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.base_url}/health")
            return bool(response.status_code == 200)
        except httpx.HTTPError:
            return False


class LocalLLM:
    """Minimal client for llama-server's OpenAI-compatible /v1/chat/completions."""

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = client

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        if self._client is not None:
            response = await self._client.post(url, json=payload, timeout=self.timeout_s)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.post(url, json=payload)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        json_schema: dict[str, object] | None = None,
    ) -> str:
        """Single-turn completion; json_schema enables grammar-constrained output."""
        payload: dict[str, Any] = {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        data = await self._post(payload)
        content = data["choices"][0]["message"]["content"]
        return str(content)

    async def complete_typed(
        self,
        prompt: str,
        schema: type[T],
        max_tokens: int = 512,
        max_retries: int = 2,
    ) -> T:
        """Completion parsed into a pydantic model, retrying on malformed JSON.

        Raises ValueError when every attempt (1 + max_retries) is malformed;
        callers drop the item rather than crash the run.
        """
        json_schema = schema.model_json_schema()
        attempt_prompt = prompt
        last_error = ""
        for attempt in range(1 + max_retries):
            text = await self.complete(
                attempt_prompt, max_tokens=max_tokens, json_schema=json_schema
            )
            try:
                return schema.model_validate(json.loads(_strip_fences(text)))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
                logger.warning("local_llm_malformed_json", attempt=attempt, error=last_error[:200])
                attempt_prompt = (
                    f"{prompt}\n\nYour previous reply was not valid JSON for the required "
                    f"schema ({last_error[:200]}). Reply again with ONLY the JSON object."
                )
        raise ValueError(f"local model returned malformed JSON after {1 + max_retries} attempts")


def _strip_fences(text: str) -> str:
    """Tolerate models that wrap JSON in markdown code fences."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()
