"""LocalLLM client, retry-on-malformed path, and server plumbing (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import BaseModel

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.arbiter import VramArbiter
from quarry_ldr.gpu.local_llm import (
    DOWNLOAD_REMEDIATION,
    LlamaServer,
    LlamaServerError,
    LocalLLM,
    _strip_fences,
    find_gguf,
    find_server_binary,
)

BASE = "http://127.0.0.1:8555"
CHAT = f"{BASE}/v1/chat/completions"


class Verdict(BaseModel):
    relevant: bool
    confidence: float


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


async def test_complete_returns_content() -> None:
    with respx.mock:
        respx.post(CHAT).mock(return_value=httpx.Response(200, json=_chat_response("hello")))
        llm = LocalLLM(BASE)
        assert await llm.complete("hi") == "hello"


async def test_complete_sends_schema_as_response_format() -> None:
    with respx.mock:
        route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=_chat_response("{}")))
        llm = LocalLLM(BASE)
        await llm.complete("hi", json_schema={"type": "object"})
        payload = json.loads(route.calls[0].request.content)
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["schema"] == {"type": "object"}
        assert payload["temperature"] == 0.0


async def test_complete_typed_parses_valid_json() -> None:
    body = '{"relevant": true, "confidence": 0.9}'
    with respx.mock:
        respx.post(CHAT).mock(return_value=httpx.Response(200, json=_chat_response(body)))
        llm = LocalLLM(BASE)
        verdict = await llm.complete_typed("judge this", Verdict)
        assert verdict.relevant is True
        assert verdict.confidence == 0.9


async def test_complete_typed_retries_then_succeeds() -> None:
    good = '{"relevant": false, "confidence": 0.2}'
    with respx.mock:
        route = respx.post(CHAT).mock(
            side_effect=[
                httpx.Response(200, json=_chat_response("not json at all")),
                httpx.Response(200, json=_chat_response(good)),
            ]
        )
        llm = LocalLLM(BASE)
        verdict = await llm.complete_typed("judge", Verdict, max_retries=2)
        assert verdict.relevant is False
        assert route.call_count == 2
        # The retry prompt tells the model what went wrong.
        retry_payload = json.loads(route.calls[1].request.content)
        assert "not valid JSON" in retry_payload["messages"][0]["content"]


async def test_complete_typed_exhausts_retries() -> None:
    with respx.mock:
        respx.post(CHAT).mock(return_value=httpx.Response(200, json=_chat_response("{broken")))
        llm = LocalLLM(BASE)
        with pytest.raises(ValueError, match="malformed JSON after 3 attempts"):
            await llm.complete_typed("judge", Verdict, max_retries=2)


async def test_complete_typed_tolerates_markdown_fences() -> None:
    fenced = '```json\n{"relevant": true, "confidence": 1.0}\n```'
    with respx.mock:
        respx.post(CHAT).mock(return_value=httpx.Response(200, json=_chat_response(fenced)))
        llm = LocalLLM(BASE)
        verdict = await llm.complete_typed("judge", Verdict)
        assert verdict.relevant is True


def test_strip_fences_variants() -> None:
    assert _strip_fences('{"a": 1}') == '{"a": 1}'
    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'


async def test_validation_error_also_retries() -> None:
    wrong_shape = '{"relevant": "definitely", "confidence": "high"}'
    good = '{"relevant": true, "confidence": 0.7}'
    with respx.mock:
        route = respx.post(CHAT).mock(
            side_effect=[
                httpx.Response(200, json=_chat_response(wrong_shape)),
                httpx.Response(200, json=_chat_response(good)),
            ]
        )
        llm = LocalLLM(BASE)
        verdict = await llm.complete_typed("judge", Verdict)
        assert verdict.confidence == 0.7
        assert route.call_count == 2


def test_find_server_binary_missing_carries_remediation(tmp_path: Path) -> None:
    with pytest.raises(LlamaServerError) as excinfo:
        find_server_binary(tmp_path)
    assert DOWNLOAD_REMEDIATION in str(excinfo.value)


def test_find_gguf_missing_carries_remediation(tmp_path: Path) -> None:
    with pytest.raises(LlamaServerError) as excinfo:
        find_gguf(tmp_path, "model.gguf")
    assert DOWNLOAD_REMEDIATION in str(excinfo.value)


def test_find_binaries_when_present(tmp_path: Path) -> None:
    import sys

    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    binary = tmp_path / "llama.cpp" / name
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"stub")
    assert find_server_binary(tmp_path) == binary

    gguf = tmp_path / "gguf" / "model.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"stub")
    assert find_gguf(tmp_path, "model.gguf") == gguf


def test_server_command_full_offload(cfg: QuarryConfig, tmp_path: Path) -> None:
    arbiter = VramArbiter(budget_mb=6656)
    server = LlamaServer(cfg, arbiter, tmp_path)
    command = server._command(tmp_path / "llama-server.exe", tmp_path / "m.gguf")
    assert "-ngl" in command and command[command.index("-ngl") + 1] == "99"
    assert "--port" in command and command[command.index("--port") + 1] == "8555"
    assert "-c" in command and command[command.index("-c") + 1] == "8192"
    assert "--jinja" in command
    assert command[command.index("--host") + 1] == "127.0.0.1"


async def test_server_registers_with_arbiter(cfg: QuarryConfig, tmp_path: Path) -> None:
    arbiter = VramArbiter(budget_mb=6656)
    server = LlamaServer(cfg, arbiter, tmp_path)
    assert server.base_url == "http://127.0.0.1:8555"
    # Registered under "triage" with the configured footprint: acquiring it
    # without a binary present must fail with the download remediation, which
    # proves the loader is wired through the arbiter.
    with pytest.raises(LlamaServerError) as excinfo:
        async with arbiter.acquire("triage"):
            pass
    assert DOWNLOAD_REMEDIATION in str(excinfo.value)
    assert arbiter.resident_models() == []


async def test_is_healthy_false_when_down() -> None:
    with respx.mock:
        respx.get(f"{BASE}/health").mock(side_effect=httpx.ConnectError("down"))
        arbiter = VramArbiter(budget_mb=6656)
        server = LlamaServer(QuarryConfig(_env_file=None), arbiter, Path("nowhere"))
        assert await server.is_healthy() is False
