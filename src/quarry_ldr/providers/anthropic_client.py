"""Anthropic API client: retries, prompt caching, batch, ledger hooks.

Rules enforced here, not merely documented:
  * The ledger is updated from the API ``usage`` block on every call.
  * Cached-corpus calls assert a byte-identical prefix by hash; drift raises
    CachePrefixError instead of silently paying full price.
  * 429/529 get exponential backoff with jitter up to api.max_retries.

Implemented in M7.
"""

from __future__ import annotations

import hashlib
from typing import TypeVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from quarry_ldr.config import QuarryConfig
from quarry_ldr.ledger import Ledger, TokenUsage

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        call must match or CachePrefixError is raised.
        """
        raise NotImplementedError

    async def batch_complete(self, requests: list[BatchRequest]) -> dict[str, CompletionResult]:
        """Batch API path (50 percent discount, stacks with caching); polls
        until done, records usage per result. Keyed by custom_id."""
        raise NotImplementedError
