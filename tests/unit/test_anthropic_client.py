"""Provider behavior against transport-level mocks and recorded shapes:
retries, ledger from usage blocks, cache prefix pinning, batch discounting."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from anthropic import AsyncAnthropic, RateLimitError

from quarry_ldr.config import QuarryConfig
from quarry_ldr.ledger import CostCapExceeded, Ledger
from quarry_ldr.pipeline.gap import GapAnalysis
from quarry_ldr.providers.anthropic_client import (
    AnthropicProvider,
    BatchRequest,
    CachePrefixError,
    MissingApiKeyError,
    hash_corpus,
)

API = "https://api.anthropic.com"
MESSAGES = f"{API}/v1/messages"
BATCHES = f"{API}/v1/messages/batches"


def _fixture(fixtures_dir: Path, name: str) -> dict:
    return json.loads((fixtures_dir / "anthropic" / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture()
def fast_cfg(cfg: QuarryConfig) -> QuarryConfig:
    cfg.api.retry_base_s = 0.001
    cfg.api.batch_poll_s = 0.001
    return cfg


@pytest.fixture()
def provider(fast_cfg: QuarryConfig) -> AnthropicProvider:
    client = AsyncAnthropic(api_key="test-key-not-real", max_retries=0)
    return AnthropicProvider(fast_cfg, Ledger(cost_cap_usd=10.0), client=client)


async def test_complete_returns_text_and_records_usage(
    provider: AnthropicProvider, fixtures_dir: Path
) -> None:
    body = _fixture(fixtures_dir, "plan_response")
    with respx.mock:
        respx.post(MESSAGES).mock(return_value=httpx.Response(200, json=body))
        result = await provider.complete(model="claude-opus-5", prompt="plan this", stage="plan")
    assert "sub_questions" in result.text
    assert result.usage.input_tokens == 812
    assert result.usage.output_tokens == 1290
    entry = provider.ledger.entries[0]
    assert entry.stage == "plan"
    # 812 * $5/MTok + 1290 * $25/MTok = 0.00406 + 0.03225
    assert entry.cost_usd == pytest.approx(0.03631)


async def test_retries_on_429_then_succeeds(
    provider: AnthropicProvider, fixtures_dir: Path
) -> None:
    rate_limited = _fixture(fixtures_dir, "rate_limit_error")
    ok = _fixture(fixtures_dir, "plan_response")
    with respx.mock:
        route = respx.post(MESSAGES).mock(
            side_effect=[
                httpx.Response(429, json=rate_limited),
                httpx.Response(529, json=rate_limited),
                httpx.Response(200, json=ok),
            ]
        )
        result = await provider.complete(model="claude-opus-5", prompt="p", stage="plan")
    assert route.call_count == 3
    assert result.usage.input_tokens == 812
    assert len(provider.ledger.entries) == 1  # failed attempts record nothing


async def test_retries_exhausted_raises(fast_cfg: QuarryConfig, fixtures_dir: Path) -> None:
    fast_cfg.api.max_retries = 1
    provider = AnthropicProvider(
        fast_cfg,
        Ledger(),
        client=AsyncAnthropic(api_key="test-key-not-real", max_retries=0),
    )
    rate_limited = _fixture(fixtures_dir, "rate_limit_error")
    with respx.mock:
        route = respx.post(MESSAGES).mock(return_value=httpx.Response(429, json=rate_limited))
        with pytest.raises(RateLimitError):
            await provider.complete(model="claude-opus-5", prompt="p")
        assert route.call_count == 2  # first try + one retry
    assert provider.ledger.entries == []


async def test_cached_corpus_sends_cache_control_and_records_write(
    provider: AnthropicProvider, fixtures_dir: Path
) -> None:
    body = _fixture(fixtures_dir, "synth_cache_write")
    with respx.mock:
        route = respx.post(MESSAGES).mock(return_value=httpx.Response(200, json=body))
        result = await provider.complete_with_cached_corpus(
            model="claude-opus-5",
            cache_name="synthesis",
            corpus="EVIDENCE CORPUS",
            brief="write section 1",
            stage="synthesize",
        )
        payload = json.loads(route.calls[0].request.content)
    content = payload["messages"][0]["content"]
    assert content[0]["text"] == "EVIDENCE CORPUS"
    assert content[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert content[1] == {"type": "text", "text": "write section 1"}
    assert result.usage.cache_creation_input_tokens == 60000
    # 60K cache write at $10/MTok = $0.60 plus 220 in + 640 out.
    assert provider.ledger.total_cost_usd == pytest.approx(0.60 + 0.0011 + 0.016)


async def test_cache_reads_register_in_usage(
    provider: AnthropicProvider, fixtures_dir: Path
) -> None:
    """The spec's required proof: cache hits show up in
    usage.cache_read_input_tokens and are priced at the cache-read rate."""
    write = _fixture(fixtures_dir, "synth_cache_write")
    read = _fixture(fixtures_dir, "synth_cache_read")
    with respx.mock:
        respx.post(MESSAGES).mock(
            side_effect=[
                httpx.Response(200, json=write),
                httpx.Response(200, json=read),
            ]
        )
        first = await provider.complete_with_cached_corpus(
            model="claude-opus-5", cache_name="s", corpus="C", brief="section 1"
        )
        second = await provider.complete_with_cached_corpus(
            model="claude-opus-5", cache_name="s", corpus="C", brief="section 2"
        )
    assert first.usage.cache_creation_input_tokens == 60000
    assert second.usage.cache_read_input_tokens == 60000
    read_entry = provider.ledger.entries[1]
    # 60K cache reads at $0.50/MTok = $0.03 plus 214 in + 588 out.
    assert read_entry.cost_usd == pytest.approx(0.03 + 214 * 5e-6 + 588 * 25e-6)


async def test_cache_prefix_drift_raises_without_network(
    provider: AnthropicProvider, fixtures_dir: Path
) -> None:
    body = _fixture(fixtures_dir, "synth_cache_write")
    with respx.mock:
        route = respx.post(MESSAGES).mock(return_value=httpx.Response(200, json=body))
        await provider.complete_with_cached_corpus(
            model="claude-opus-5", cache_name="s", corpus="CORPUS A", brief="b1"
        )
        with pytest.raises(CachePrefixError, match="refusing to pay"):
            await provider.complete_with_cached_corpus(
                model="claude-opus-5", cache_name="s", corpus="CORPUS B", brief="b2"
            )
        assert route.call_count == 1  # the drifted call never reached the API
    assert hash_corpus("CORPUS A") != hash_corpus("CORPUS B")


async def test_distinct_cache_names_are_independent(
    provider: AnthropicProvider, fixtures_dir: Path
) -> None:
    body = _fixture(fixtures_dir, "synth_cache_write")
    with respx.mock:
        respx.post(MESSAGES).mock(return_value=httpx.Response(200, json=body))
        await provider.complete_with_cached_corpus(
            model="claude-opus-5", cache_name="run1", corpus="A", brief="b"
        )
        await provider.complete_with_cached_corpus(
            model="claude-opus-5", cache_name="run2", corpus="B", brief="b"
        )  # different name, different corpus: fine


async def test_complete_typed_parses_gap_fixture(
    provider: AnthropicProvider, fixtures_dir: Path
) -> None:
    body = _fixture(fixtures_dir, "gap_response")
    with respx.mock:
        respx.post(MESSAGES).mock(return_value=httpx.Response(200, json=body))
        gap = await provider.complete_typed(
            model="claude-sonnet-5", prompt="analyze", schema=GapAnalysis, stage="gap"
        )
    assert gap.saturated is False
    assert gap.new_queries == ["vesterholm pilot independent efficiency audit"]
    assert gap.assessments[1].covered is False


async def test_complete_typed_reprompts_once_on_malformed(
    provider: AnthropicProvider, fixtures_dir: Path
) -> None:
    bad = _fixture(fixtures_dir, "plan_response")
    bad = json.loads(json.dumps(bad))
    bad["content"] = [{"type": "text", "text": "not json"}]
    good = _fixture(fixtures_dir, "gap_response")
    with respx.mock:
        route = respx.post(MESSAGES).mock(
            side_effect=[httpx.Response(200, json=bad), httpx.Response(200, json=good)]
        )
        gap = await provider.complete_typed(
            model="claude-sonnet-5", prompt="analyze", schema=GapAnalysis
        )
        retry_payload = json.loads(route.calls[1].request.content)
    assert gap.rationale == "Coverage is thin on sq02."
    assert "not valid JSON" in retry_payload["messages"][0]["content"]
    assert route.call_count == 2


async def test_complete_typed_gives_up_after_two_attempts(
    provider: AnthropicProvider, fixtures_dir: Path
) -> None:
    bad = _fixture(fixtures_dir, "plan_response")
    bad = json.loads(json.dumps(bad))
    bad["content"] = [{"type": "text", "text": "still not json"}]
    with respx.mock:
        respx.post(MESSAGES).mock(return_value=httpx.Response(200, json=bad))
        with pytest.raises(ValueError, match="malformed JSON twice"):
            await provider.complete_typed(
                model="claude-sonnet-5", prompt="analyze", schema=GapAnalysis
            )


async def test_missing_api_key_error(fast_cfg: QuarryConfig) -> None:
    provider = AnthropicProvider(fast_cfg, Ledger())
    with pytest.raises(MissingApiKeyError, match=r"\.env\.example"):
        await provider.complete(model="claude-opus-5", prompt="p")


async def test_cost_cap_propagates(fast_cfg: QuarryConfig, fixtures_dir: Path) -> None:
    provider = AnthropicProvider(
        fast_cfg,
        Ledger(cost_cap_usd=0.01),
        client=AsyncAnthropic(api_key="test-key-not-real", max_retries=0),
    )
    body = _fixture(fixtures_dir, "plan_response")  # costs ~$0.036
    with respx.mock:
        respx.post(MESSAGES).mock(return_value=httpx.Response(200, json=body))
        with pytest.raises(CostCapExceeded):
            await provider.complete(model="claude-opus-5", prompt="p", stage="plan")
    assert len(provider.ledger.entries) == 1  # the crossing entry is recorded


def _batch_object(status: str) -> dict:
    return {
        "id": "msgbatch_fixture01",
        "type": "message_batch",
        "processing_status": status,
        "request_counts": {
            "processing": 0 if status == "ended" else 2,
            "succeeded": 2 if status == "ended" else 0,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
        },
        "created_at": "2026-08-03T00:00:00Z",
        "expires_at": "2026-08-04T00:00:00Z",
        "ended_at": "2026-08-03T00:10:00Z" if status == "ended" else None,
        "archived_at": None,
        "cancel_initiated_at": None,
        "results_url": f"{BATCHES}/msgbatch_fixture01/results" if status == "ended" else None,
    }


async def test_batch_complete_records_half_price(
    provider: AnthropicProvider, fixtures_dir: Path
) -> None:
    message = _fixture(fixtures_dir, "plan_response")
    results_jsonl = "\n".join(
        json.dumps({"custom_id": cid, "result": {"type": "succeeded", "message": message}})
        for cid in ("a", "b")
    )
    with respx.mock:
        respx.post(BATCHES).mock(
            return_value=httpx.Response(200, json=_batch_object("in_progress"))
        )
        respx.get(f"{BATCHES}/msgbatch_fixture01").mock(
            return_value=httpx.Response(200, json=_batch_object("ended"))
        )
        respx.get(f"{BATCHES}/msgbatch_fixture01/results").mock(
            return_value=httpx.Response(
                200, content=results_jsonl, headers={"content-type": "application/x-jsonl"}
            )
        )
        results = await provider.batch_complete(
            [
                BatchRequest(custom_id="a", model="claude-haiku-4-5-20251001", prompt="p1"),
                BatchRequest(custom_id="b", model="claude-haiku-4-5-20251001", prompt="p2"),
            ],
            stage="extract",
        )
    assert set(results) == {"a", "b"}
    for entry in provider.ledger.entries:
        assert entry.batch is True
        # Haiku batch: (812 * $1 + 1290 * $5) / 1e6 * 0.5
        assert entry.cost_usd == pytest.approx((812 * 1 + 1290 * 5) / 1e6 * 0.5)
