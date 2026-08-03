"""Anthropic API client: retries, prompt caching, batch, ledger hooks.

Rules enforced here, not merely documented:
  * The ledger is updated from the API ``usage`` block on every call.
  * Cached-corpus calls assert a byte-identical prefix by hash; drift raises
    CachePrefixError instead of silently paying full price.
  * Rate limits and overload (429/5xx) get exponential backoff with jitter up
    to api.max_retries; our layer owns retries (the SDK's are disabled).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import AsyncIterator
from typing import Any, TypeVar, cast

import anthropic
from anthropic import NOT_GIVEN, AsyncAnthropic
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from quarry_ldr.config import QuarryConfig
from quarry_ldr.ledger import Ledger, TokenUsage
from quarry_ldr.logging import get_logger

logger = get_logger(component="anthropic")

T = TypeVar("T", bound=BaseModel)


class CompletionResult(BaseModel):
    text: str
    usage: TokenUsage
    model: str
    stop_reason: str | None = None


class CachePrefixError(Exception):
    """A cached-corpus call tried to send a different prefix than the one the
    cache was primed with. This would silently cost full input price."""


class MissingApiKeyError(Exception):
    def __init__(self) -> None:
        super().__init__(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and paste "
            "your key. Tests and fixture runs never need it."
        )


def hash_corpus(text: str) -> str:
    """SHA-256 of the exact corpus bytes; the cache-prefix identity."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class BatchRequest(BaseModel):
    custom_id: str
    model: str
    prompt: str
    system: str | None = None
    max_tokens: int = 1024


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


class AnthropicProvider:
    """All Anthropic traffic flows through this one class.

    ``client`` is injectable so tests mock at the transport layer.
    """

    def __init__(
        self,
        cfg: QuarryConfig,
        ledger: Ledger,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self.cfg = cfg
        self.ledger = ledger
        self._client = client
        self._corpus_hashes: dict[str, str] = {}  # logical cache name -> pinned hash

    def _get_client(self) -> AsyncAnthropic:
        if self._client is None:
            api_key = self.cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise MissingApiKeyError()
            # max_retries=0: this class owns retry policy so backoff is
            # observable and configurable, not hidden in the SDK.
            self._client = AsyncAnthropic(api_key=api_key, max_retries=0)
        return self._client

    async def _create(self, **kwargs: Any) -> anthropic.types.Message:
        client = self._get_client()
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential_jitter(initial=self.cfg.api.retry_base_s, max=60),
            stop=stop_after_attempt(self.cfg.api.max_retries + 1),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    logger.warning("api_retry", attempt=attempt.retry_state.attempt_number)
                return await client.messages.create(**kwargs)
        raise RuntimeError("unreachable")  # pragma: no cover

    def _record(
        self,
        message: anthropic.types.Message,
        model: str,
        stage: str,
        iteration: int,
        batch: bool = False,
    ) -> TokenUsage:
        usage = TokenUsage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            cache_creation_input_tokens=message.usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=message.usage.cache_read_input_tokens or 0,
        )
        self.ledger.record(model=model, usage=usage, stage=stage, iteration=iteration, batch=batch)
        return usage

    @staticmethod
    def _text_of(message: anthropic.types.Message) -> str:
        return "".join(block.text for block in message.content if block.type == "text")

    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        stage: str = "",
        iteration: int = 0,
    ) -> CompletionResult:
        """Plain completion with retries; records usage in the ledger."""
        message = await self._create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system if system is not None else NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = self._record(message, model, stage, iteration)
        return CompletionResult(
            text=self._text_of(message),
            usage=usage,
            model=model,
            stop_reason=message.stop_reason,
        )

    async def complete_typed(
        self,
        *,
        model: str,
        prompt: str,
        schema: type[T],
        system: str | None = None,
        max_tokens: int = 4096,
        stage: str = "",
        iteration: int = 0,
    ) -> T:
        """Completion parsed as JSON into ``schema``; one reprompt on malformed."""
        attempt_prompt = prompt
        last_error = ""
        for _ in range(2):
            result = await self.complete(
                model=model,
                prompt=attempt_prompt,
                system=system,
                max_tokens=max_tokens,
                stage=stage,
                iteration=iteration,
            )
            try:
                return schema.model_validate(json.loads(_strip_fences(result.text)))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
                logger.warning("api_malformed_json", stage=stage, error=last_error[:200])
                attempt_prompt = (
                    f"{prompt}\n\nYour previous reply was not valid JSON for the "
                    f"required schema ({last_error[:300]}). Reply again with ONLY "
                    "the JSON object, no prose, no code fences."
                )
        raise ValueError(f"model returned malformed JSON twice for stage {stage}: {last_error}")

    async def complete_with_cached_corpus(
        self,
        *,
        model: str,
        cache_name: str,
        corpus: str,
        brief: str,
        system: str | None = None,
        max_tokens: int = 4096,
        stage: str = "",
        iteration: int = 0,
    ) -> CompletionResult:
        """Two-block message: [corpus with cache_control 1h, brief].

        The first call under ``cache_name`` pins hash(corpus); every later
        call must match byte for byte or CachePrefixError is raised, because
        a drifted prefix would silently pay full input price.
        """
        corpus_hash = hash_corpus(corpus)
        pinned = self._corpus_hashes.get(cache_name)
        if pinned is None:
            self._corpus_hashes[cache_name] = corpus_hash
        elif pinned != corpus_hash:
            raise CachePrefixError(
                f"corpus for cache {cache_name!r} changed (pinned {pinned[:12]}, "
                f"got {corpus_hash[:12]}); refusing to pay full price silently"
            )
        message = await self._create(
            model=model,
            max_tokens=max_tokens,
            system=system if system is not None else NOT_GIVEN,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": corpus,
                            "cache_control": {
                                "type": "ephemeral",
                                "ttl": self.cfg.api.cache_ttl,
                            },
                        },
                        {"type": "text", "text": brief},
                    ],
                }
            ],
        )
        usage = self._record(message, model, stage, iteration)
        logger.info(
            "cached_corpus_call",
            cache_name=cache_name,
            cache_write=usage.cache_creation_input_tokens,
            cache_read=usage.cache_read_input_tokens,
        )
        return CompletionResult(
            text=self._text_of(message),
            usage=usage,
            model=model,
            stop_reason=message.stop_reason,
        )

    async def batch_complete(
        self, requests: list[BatchRequest], stage: str = "", iteration: int = 0
    ) -> dict[str, CompletionResult]:
        """Batch API path (50 percent discount, stacks with caching); polls
        until done, records usage per result with batch=True. Keyed by custom_id."""
        client = self._get_client()
        payload: list[dict[str, Any]] = []
        for request in requests:
            params: dict[str, Any] = {
                "model": request.model,
                "max_tokens": request.max_tokens,
                "messages": [{"role": "user", "content": request.prompt}],
            }
            if request.system:
                params["system"] = request.system
            payload.append({"custom_id": request.custom_id, "params": params})
        batch = await client.messages.batches.create(requests=cast(Any, payload))
        import asyncio

        while batch.processing_status != "ended":
            await asyncio.sleep(self.cfg.api.batch_poll_s)
            batch = await client.messages.batches.retrieve(batch.id)

        models_by_id = {request.custom_id: request.model for request in requests}
        results: dict[str, CompletionResult] = {}
        entries: AsyncIterator[Any] = await client.messages.batches.results(batch.id)
        async for entry in entries:
            if entry.result.type != "succeeded":
                logger.warning(
                    "batch_entry_failed", custom_id=entry.custom_id, kind=entry.result.type
                )
                continue
            message = entry.result.message
            model = models_by_id.get(entry.custom_id, message.model)
            usage = self._record(message, model, stage, iteration, batch=True)
            results[entry.custom_id] = CompletionResult(
                text=self._text_of(message),
                usage=usage,
                model=model,
                stop_reason=message.stop_reason,
            )
        return results
