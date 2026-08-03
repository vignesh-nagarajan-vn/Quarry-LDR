"""Token and cost accounting from API ``usage`` blocks. Never estimated.

Pricing is date-aware because Sonnet 5 is on introductory pricing ($2/$10)
through 2026-08-31, after which it becomes $3/$15. Costs are always computed
from the ``usage`` block the API returned; Claude 4.7+ tokenizers produce
roughly 30 percent more tokens for the same text than older ones, so
character-count estimates are banned by design.

Implemented in M1.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Mirror of the Anthropic API usage block."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class ModelPrice(BaseModel):
    """USD per million tokens. ``batch`` runs halve every component (discounts stack)."""

    input: float
    output: float
    cache_write_1h: float
    cache_read: float


# (effective_from, price): the entry with the latest effective_from <= run date wins.
PRICING: dict[str, list[tuple[date, ModelPrice]]] = {
    "claude-opus-5": [
        (date(2025, 1, 1), ModelPrice(input=5.0, output=25.0, cache_write_1h=10.0, cache_read=0.5)),
    ],
    "claude-sonnet-5": [
        # Introductory pricing through 2026-08-31.
        (date(2025, 1, 1), ModelPrice(input=2.0, output=10.0, cache_write_1h=4.0, cache_read=0.2)),
        (date(2026, 9, 1), ModelPrice(input=3.0, output=15.0, cache_write_1h=6.0, cache_read=0.3)),
    ],
    "claude-haiku-4-5-20251001": [
        (date(2025, 1, 1), ModelPrice(input=1.0, output=5.0, cache_write_1h=2.0, cache_read=0.1)),
    ],
}

BATCH_DISCOUNT = 0.5


class UnknownModelError(Exception):
    """Raised when a cost is recorded for a model with no pricing entry."""


class CostCapExceeded(Exception):
    """Raised by :meth:`Ledger.record` when the run cost cap is crossed."""

    def __init__(self, total_usd: float, cap_usd: float) -> None:
        self.total_usd = total_usd
        self.cap_usd = cap_usd
        super().__init__(f"run cost ${total_usd:.4f} exceeded cap ${cap_usd:.2f}")


class LedgerEntry(BaseModel):
    timestamp: datetime
    model: str
    stage: str
    iteration: int = 0
    batch: bool = False
    usage: TokenUsage
    cost_usd: float
    context: str = ""


class LedgerSummary(BaseModel):
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_cache_write_tokens: int
    total_cache_read_tokens: int
    by_model: dict[str, float] = Field(default_factory=dict)
    by_stage: dict[str, float] = Field(default_factory=dict)
    by_iteration: dict[int, float] = Field(default_factory=dict)


def price_for(model: str, on: date) -> ModelPrice:
    """Resolve the price row effective for ``model`` on date ``on``."""
    raise NotImplementedError


def compute_cost(usage: TokenUsage, price: ModelPrice, batch: bool = False) -> float:
    """Cost in USD from a usage block. Batch halves every component."""
    raise NotImplementedError


class Ledger:
    """In-memory ledger for one run; the orchestrator persists entries via RunStore."""

    def __init__(self, cost_cap_usd: float | None = None) -> None:
        self.cost_cap_usd = cost_cap_usd
        self._entries: list[LedgerEntry] = []

    def record(
        self,
        model: str,
        usage: TokenUsage,
        stage: str,
        iteration: int = 0,
        batch: bool = False,
        context: str = "",
        on: date | None = None,
    ) -> LedgerEntry:
        """Compute cost from ``usage``, append an entry, enforce the cap.

        Raises :class:`CostCapExceeded` after appending, so the entry that
        crossed the cap is still accounted for.
        """
        raise NotImplementedError

    @property
    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    @property
    def total_cost_usd(self) -> float:
        raise NotImplementedError

    def summary(self) -> LedgerSummary:
        raise NotImplementedError

    def to_markdown(self) -> str:
        """Render the cost ledger table appended to every report."""
        raise NotImplementedError

    def dump(self) -> list[dict[str, object]]:
        """JSON-safe entries for persistence."""
        raise NotImplementedError

    @classmethod
    def load(cls, entries: list[dict[str, object]], cost_cap_usd: float | None = None) -> Ledger:
        """Rehydrate from persisted entries (used by resume)."""
        raise NotImplementedError
