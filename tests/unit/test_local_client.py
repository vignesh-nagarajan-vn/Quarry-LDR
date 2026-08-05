"""LocalProvider: protocol conformance, $0 ledger recording from server
usage blocks, grammar-constrained typed calls, and cache-prefix pinning."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.local_llm import LocalLLM
from quarry_ldr.ledger import Ledger
from quarry_ldr.providers.anthropic_client import AnthropicProvider
from quarry_ldr.providers.base import CachePrefixError, Provider
from quarry_ldr.providers.local_client import LocalProvider

BASE = "http://127.0.0.1:8555"
CHAT = f"{BASE}/v1/chat/completions"
MODEL = "local/test-model.gguf"


class Verdict(BaseModel):
    relevant: bool
    confidence: float


def _chat_response(
    content: str,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
    include_usage: bool = True,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ]
    }
    if include_usage:
        body["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    return body


def _provider(ledger: Ledger | None = None) -> LocalProvider:
    return LocalProvider(ledger if ledger is not None else Ledger(), LocalLLM(BASE), MODEL)


def test_conforms_to_provider_protocol(cfg: QuarryConfig) -> None:
    # mypy checks these assignments; drift in either concrete class's shape
    # versus the Protocol fails type checking, not just this assertion.
    local: Provider = _provider()
    api: Provider = AnthropicProvider(cfg, Ledger())
    assert local is not None
    assert api is not None


def test_model_name_must_carry_local_prefix() -> None:
    with pytest.raises(ValueError, match="local/"):
        LocalProvider(Ledger(), LocalLLM(BASE), "qwen-8b.gguf")


async def test_complete_records_usage_at_zero_cost() -> None:
    ledger = Ledger()
    with respx.mock:
        respx.post(CHAT).mock(return_value=httpx.Response(200, json=_chat_response("hello", 42, 7)))
        result = await _provider(ledger).complete(
            model="claude-opus-5", prompt="hi", stage="plan", iteration=3
        )
    assert result.text == "hello"
    assert result.model == MODEL
    assert result.stop_reason == "end_turn"
    entry = ledger.entries[0]
    assert entry.model == MODEL
    assert entry.stage == "plan"
    assert entry.iteration == 3
    assert entry.usage.input_tokens == 42
    assert entry.usage.output_tokens == 7
    assert entry.cost_usd == 0.0
    assert ledger.total_cost_usd == 0.0


async def test_finish_reason_length_maps_to_max_tokens() -> None:
    with respx.mock:
        respx.post(CHAT).mock(
            return_value=httpx.Response(200, json=_chat_response("txt", finish_reason="length"))
        )
        result = await _provider().complete(model="m", prompt="p")
    assert result.stop_reason == "max_tokens"


async def test_missing_usage_block_records_zeros() -> None:
    ledger = Ledger()
    with respx.mock:
        respx.post(CHAT).mock(
            return_value=httpx.Response(200, json=_chat_response("ok", include_usage=False))
        )
        await _provider(ledger).complete(model="m", prompt="p", stage="plan")
    assert ledger.entries[0].usage.input_tokens == 0
    assert ledger.entries[0].usage.output_tokens == 0
    assert ledger.entries[0].cost_usd == 0.0


async def test_system_prompt_becomes_system_message() -> None:
    with respx.mock:
        route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=_chat_response("ok")))
        await _provider().complete(model="m", prompt="question", system="be terse")
        payload = json.loads(route.calls[0].request.content)
    assert payload["messages"][0] == {"role": "system", "content": "be terse"}
    assert payload["messages"][1]["content"] == "question"


async def test_complete_typed_constrains_and_records_every_attempt() -> None:
    ledger = Ledger()
    good = '{"relevant": true, "confidence": 0.9}'
    with respx.mock:
        route = respx.post(CHAT).mock(
            side_effect=[
                httpx.Response(200, json=_chat_response("not json")),
                httpx.Response(200, json=_chat_response(good)),
            ]
        )
        verdict = await _provider(ledger).complete_typed(
            model="m", prompt="judge", schema=Verdict, stage="gap"
        )
    assert verdict.relevant is True
    payload = json.loads(route.calls[0].request.content)
    assert payload["response_format"]["type"] == "json_schema"
    # The failed attempt's tokens were really generated: two $0 entries.
    assert len(ledger.entries) == 2
    assert all(entry.cost_usd == 0.0 for entry in ledger.entries)
    assert all(entry.stage == "gap" for entry in ledger.entries)


async def test_complete_typed_exhausts_retries() -> None:
    ledger = Ledger()
    with respx.mock:
        respx.post(CHAT).mock(return_value=httpx.Response(200, json=_chat_response("{broken")))
        with pytest.raises(ValueError, match="after 3 attempts"):
            await _provider(ledger).complete_typed(model="m", prompt="judge", schema=Verdict)
    assert len(ledger.entries) == 3


async def test_cached_corpus_concatenates_and_pins() -> None:
    ledger = Ledger()
    provider = _provider(ledger)
    with respx.mock:
        route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=_chat_response("sec")))
        first = await provider.complete_with_cached_corpus(
            model="m",
            cache_name="synthesis:abc",
            corpus="CORPUS",
            brief="BRIEF",
            stage="synthesize",
        )
        await provider.complete_with_cached_corpus(
            model="m", cache_name="synthesis:abc", corpus="CORPUS", brief="OTHER BRIEF"
        )
        payload = json.loads(route.calls[0].request.content)
    assert first.text == "sec"
    assert payload["messages"][0]["content"] == "CORPUS\n\nBRIEF"
    assert len(ledger.entries) == 2


async def test_cached_corpus_drift_raises() -> None:
    provider = _provider()
    with respx.mock:
        respx.post(CHAT).mock(return_value=httpx.Response(200, json=_chat_response("sec")))
        await provider.complete_with_cached_corpus(
            model="m", cache_name="c", corpus="CORPUS A", brief="b"
        )
        with pytest.raises(CachePrefixError, match="citation numbering"):
            await provider.complete_with_cached_corpus(
                model="m", cache_name="c", corpus="CORPUS B", brief="b"
            )
