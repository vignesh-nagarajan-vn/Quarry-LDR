"""Local provider: the ``Provider`` contract over a llama-server client.

PLAN, GAP, and SYNTHESIZE route through this class when ``engine.mode`` is
local or assisted. The API client's rules transfer where they have meaning:

  * Every call records real token counts from the server's ``usage`` block
    in the ledger, priced at zero under a ``local/<gguf>`` model id.
  * Cached-corpus calls keep the byte-identical-prefix hash pin. Locally
    the stake is not dollars: a drifted corpus would break the citation
    numbering sections were written against and force a full KV recompute.
  * ``complete_typed`` uses grammar-constrained decoding (``json_schema``),
    which llama-server enforces natively, with the same corrective-reprompt
    retry loop the triage client uses.

The ``model=`` argument call sites pass (an API model id from config) is
accepted for protocol compatibility but not routed; the wrapped server runs
exactly one model, named at construction.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from quarry_ldr.gpu.local_llm import LocalLLM, _strip_fences
from quarry_ldr.ledger import LOCAL_MODEL_PREFIX, Ledger, TokenUsage
from quarry_ldr.logging import get_logger
from quarry_ldr.providers.base import CachePrefixError, CompletionResult, hash_corpus

logger = get_logger(component="local_provider")

T = TypeVar("T", bound=BaseModel)

# llama-server speaks OpenAI finish reasons; the pipeline speaks Anthropic
# stop reasons (empty-section retries key off "max_tokens").
_FINISH_TO_STOP = {"stop": "end_turn", "length": "max_tokens"}


class LocalProvider:
    """All local reasoning traffic flows through this one class."""

    def __init__(
        self,
        ledger: Ledger,
        local_llm: LocalLLM,
        model_name: str,
        max_retries: int = 2,
    ) -> None:
        if not model_name.startswith(LOCAL_MODEL_PREFIX):
            raise ValueError(
                f"model_name must start with {LOCAL_MODEL_PREFIX!r} so the ledger "
                f"prices it at zero; got {model_name!r}"
            )
        self.ledger = ledger
        self.local_llm = local_llm
        self.model_name = model_name
        self.max_retries = max_retries
        self._corpus_hashes: dict[str, str] = {}  # logical cache name -> pinned hash

    def _record(self, usage: TokenUsage, stage: str, iteration: int) -> None:
        self.ledger.record(model=self.model_name, usage=usage, stage=stage, iteration=iteration)

    @staticmethod
    def _stop_reason(finish_reason: str | None) -> str | None:
        if finish_reason is None:
            return None
        return _FINISH_TO_STOP.get(finish_reason, finish_reason)

    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        stage: str = "",
        iteration: int = 0,
    ) -> CompletionResult:
        """Plain completion; records the server usage block in the ledger."""
        logger.debug("local_request", requested_model=model, actual=self.model_name, stage=stage)
        text, usage, finish_reason = await self.local_llm.complete_with_usage(
            prompt, max_tokens=max_tokens, system=system
        )
        self._record(usage, stage, iteration)
        return CompletionResult(
            text=text,
            usage=usage,
            model=self.model_name,
            stop_reason=self._stop_reason(finish_reason),
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
        """Grammar-constrained completion parsed into ``schema``.

        Records every attempt's usage (the tokens were really generated);
        raises ValueError when all 1 + max_retries attempts are malformed.
        """
        logger.debug("local_request", requested_model=model, actual=self.model_name, stage=stage)
        json_schema = schema.model_json_schema()
        attempt_prompt = prompt
        last_error = ""
        for attempt in range(1 + self.max_retries):
            text, usage, _ = await self.local_llm.complete_with_usage(
                attempt_prompt, max_tokens=max_tokens, json_schema=json_schema, system=system
            )
            self._record(usage, stage, iteration)
            try:
                return schema.model_validate(json.loads(_strip_fences(text)))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
                logger.warning(
                    "local_malformed_json", stage=stage, attempt=attempt, error=last_error[:200]
                )
                attempt_prompt = (
                    f"{prompt}\n\nYour previous reply was not valid JSON for the required "
                    f"schema ({last_error[:200]}). Reply again with ONLY the JSON object."
                )
        raise ValueError(
            f"local model returned malformed JSON for stage {stage!r} after "
            f"{1 + self.max_retries} attempts: {last_error[:200]}"
        )

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
        """Corpus and brief concatenated into one prompt, hash pin enforced.

        llama-server reuses its KV cache when consecutive requests share a
        byte-identical prefix; the pin proves every call under ``cache_name``
        really sent the same corpus, which is also what keeps citation
        numbering coherent across section calls.
        """
        corpus_hash = hash_corpus(corpus)
        pinned = self._corpus_hashes.get(cache_name)
        if pinned is None:
            self._corpus_hashes[cache_name] = corpus_hash
        elif pinned != corpus_hash:
            raise CachePrefixError(
                f"corpus for cache {cache_name!r} changed (pinned {pinned[:12]}, "
                f"got {corpus_hash[:12]}); a drifted prefix breaks citation "
                "numbering and forces a full KV recompute"
            )
        logger.debug("local_request", requested_model=model, actual=self.model_name, stage=stage)
        text, usage, finish_reason = await self.local_llm.complete_with_usage(
            f"{corpus}\n\n{brief}", max_tokens=max_tokens, system=system
        )
        self._record(usage, stage, iteration)
        return CompletionResult(
            text=text,
            usage=usage,
            model=self.model_name,
            stop_reason=self._stop_reason(finish_reason),
        )
