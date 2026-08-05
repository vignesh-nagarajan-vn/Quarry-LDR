"""Provider contract shared by the API client and the local client.

v1 amended the original single-provider design: PLAN, GAP, and SYNTHESIZE
type against the ``Provider`` protocol and the orchestrator picks the
concrete backend per ``engine.mode``. The pieces both backends must agree
on live here: the result model, the cache-prefix identity and its drift
error, and the protocol itself.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, TypeVar

from pydantic import BaseModel

from quarry_ldr.ledger import TokenUsage

T = TypeVar("T", bound=BaseModel)


class CompletionResult(BaseModel):
    text: str
    usage: TokenUsage
    model: str
    stop_reason: str | None = None


class CachePrefixError(Exception):
    """A cached-corpus call tried to send a different prefix than the one the
    cache was primed with. On the API this silently costs full input price;
    locally it silently recomputes the whole KV prefix and can desync the
    corpus that citation numbering was assigned against."""


def hash_corpus(text: str) -> str:
    """SHA-256 of the exact corpus bytes; the cache-prefix identity."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Provider(Protocol):
    """Structural contract every completion backend satisfies."""

    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        stage: str = "",
        iteration: int = 0,
    ) -> CompletionResult: ...

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
    ) -> T: ...

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
    ) -> CompletionResult: ...
