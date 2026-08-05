"""Live llama-server lifecycle under the arbiter, on the real GPU.

Skips (rather than fails) when the binary or GGUF has not been downloaded,
so `pytest -m gpu` stays runnable before `download_models.py`.
"""

from __future__ import annotations

import time

import pytest

from quarry_ldr.config import load_config
from quarry_ldr.gpu.arbiter import TorchCudaBackend, VramArbiter
from quarry_ldr.gpu.local_llm import (
    LlamaServer,
    LlamaServerError,
    LocalLLM,
    find_gguf,
    find_server_binary,
    synth_server_spec,
)
from quarry_ldr.logging import get_logger
from quarry_ldr.pipeline.triage import TriageVerdict

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def live_cfg():  # type: ignore[no-untyped-def]
    cfg = load_config()
    try:
        find_server_binary(cfg.run.models_dir)
        find_gguf(cfg.run.models_dir, cfg.models.triage_gguf_file)
    except LlamaServerError as exc:
        pytest.skip(f"local models not downloaded: {exc}")
    return cfg


async def test_server_lifecycle_and_typed_json(live_cfg) -> None:  # type: ignore[no-untyped-def]
    log = get_logger(component="test_local_llm_live")
    arbiter = VramArbiter(budget_mb=live_cfg.gpu.vram_budget_mb, backend=TorchCudaBackend())
    server = LlamaServer(live_cfg, arbiter, live_cfg.run.models_dir)

    started = time.monotonic()
    await server.start()
    startup_s = time.monotonic() - started
    assert await server.is_healthy()
    assert arbiter.resident_models() == ["triage"]
    resident_mb = arbiter.resident_footprint_mb()

    llm = LocalLLM(server.base_url)
    prompt = (
        "You are filtering evidence for a research question.\n"
        "Question: What temperature does the sand bed reach?\n"
        "Passage:\n---\nResistive heaters raise the sand bed to 600 degrees "
        "Celsius during off-peak hours.\n---\n"
        'Reply with JSON only, exactly this shape: {"relevant": true/false, '
        '"claim": "...", "evidence_span": "...", "confidence": 0.0-1.0}'
    )
    t0 = time.monotonic()
    verdict = await llm.complete_typed(prompt, TriageVerdict, max_tokens=256)
    latency_s = time.monotonic() - t0

    log.info(
        "live_triage_measured",
        startup_s=round(startup_s, 1),
        verdict_latency_s=round(latency_s, 2),
        resident_mb=resident_mb,
        relevant=verdict.relevant,
        confidence=verdict.confidence,
    )
    assert verdict.relevant is True
    assert "600" in verdict.evidence_span or "600" in verdict.claim

    await server.stop()
    assert arbiter.resident_models() == []
    assert not await server.is_healthy()


async def test_synth_server_lifecycle_and_generation(live_cfg) -> None:  # type: ignore[no-untyped-def]
    try:
        find_gguf(live_cfg.run.models_dir, live_cfg.models.synth_gguf_file)
    except LlamaServerError as exc:
        pytest.skip(f"synth GGUF not downloaded: {exc}")
    log = get_logger(component="test_local_llm_live")
    arbiter = VramArbiter(budget_mb=live_cfg.gpu.vram_budget_mb, backend=TorchCudaBackend())
    server = LlamaServer(
        live_cfg, arbiter, live_cfg.run.models_dir, spec=synth_server_spec(live_cfg)
    )

    started = time.monotonic()
    await server.start()
    startup_s = time.monotonic() - started
    assert await server.is_healthy()
    assert arbiter.resident_models() == ["synth"]

    llm = LocalLLM(server.base_url)
    t0 = time.monotonic()
    text, usage, finish_reason = await llm.complete_with_usage(
        "Write two sentences about sand batteries for grid storage.", max_tokens=128
    )
    latency_s = time.monotonic() - t0
    log.info(
        "live_synth_measured",
        startup_s=round(startup_s, 1),
        latency_s=round(latency_s, 2),
        output_tokens=usage.output_tokens,
        tok_s=round(usage.output_tokens / latency_s, 1) if latency_s > 0 else 0.0,
        finish_reason=finish_reason,
    )
    assert text.strip()
    assert usage.output_tokens > 0
    # reasoning_budget 0 must suppress Qwen3 thinking blocks entirely.
    assert "<think>" not in text

    await server.stop()
    assert arbiter.resident_models() == []
    assert not await server.is_healthy()
