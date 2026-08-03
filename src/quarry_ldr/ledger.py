"""Token and cost accounting from API ``usage`` blocks. Never estimated.

Pricing is date-aware because Sonnet 5 is on introductory pricing ($2/$10)
through 2026-08-31, after which it becomes $3/$15. Costs are always computed
from the ``usage`` block the API returned; Claude 4.7+ tokenizers produce
roughly 30 percent more tokens for the same text than older ones, so
character-count estimates are banned by design.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

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
    try:
        rows = PRICING[model]
    except KeyError:
        known = ", ".join(sorted(PRICING))
        raise UnknownModelError(f"no pricing for model {model!r}; known: {known}") from None
    effective = [price for effective_from, price in rows if effective_from <= on]
    # Dates before the first entry fall back to the earliest known price.
    return effective[-1] if effective else rows[0][1]


def compute_cost(usage: TokenUsage, price: ModelPrice, batch: bool = False) -> float:
    """Cost in USD from a usage block. Batch halves every component."""
    cost = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_creation_input_tokens * price.cache_write_1h
        + usage.cache_read_input_tokens * price.cache_read
    ) / 1_000_000
    return cost * BATCH_DISCOUNT if batch else cost


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
        effective_date = on if on is not None else datetime.now(UTC).date()
        price = price_for(model, effective_date)
        entry = LedgerEntry(
            timestamp=datetime.now(UTC),
            model=model,
            stage=stage,
            iteration=iteration,
            batch=batch,
            usage=usage,
            cost_usd=compute_cost(usage, price, batch=batch),
            context=context,
        )
        self._entries.append(entry)
        if self.cost_cap_usd is not None and self.total_cost_usd > self.cost_cap_usd:
            raise CostCapExceeded(self.total_cost_usd, self.cost_cap_usd)
        return entry

    @property
    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    @property
    def total_cost_usd(self) -> float:
        return sum(entry.cost_usd for entry in self._entries)

    def summary(self) -> LedgerSummary:
        by_model: dict[str, float] = {}
        by_stage: dict[str, float] = {}
        by_iteration: dict[int, float] = {}
        for entry in self._entries:
            by_model[entry.model] = by_model.get(entry.model, 0.0) + entry.cost_usd
            by_stage[entry.stage] = by_stage.get(entry.stage, 0.0) + entry.cost_usd
            by_iteration[entry.iteration] = by_iteration.get(entry.iteration, 0.0) + entry.cost_usd
        return LedgerSummary(
            total_cost_usd=self.total_cost_usd,
            total_input_tokens=sum(e.usage.input_tokens for e in self._entries),
            total_output_tokens=sum(e.usage.output_tokens for e in self._entries),
            total_cache_write_tokens=sum(
                e.usage.cache_creation_input_tokens for e in self._entries
            ),
            total_cache_read_tokens=sum(e.usage.cache_read_input_tokens for e in self._entries),
            by_model=by_model,
            by_stage=by_stage,
            by_iteration=by_iteration,
        )

    def to_markdown(self) -> str:
        """Render the cost ledger table appended to every report."""
        lines = [
            "## Cost ledger",
            "",
            "| Stage | Iter | Model | Batch | Input | Output | Cache write | Cache read | Cost |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for e in self._entries:
            lines.append(
                f"| {e.stage} | {e.iteration} | {e.model} | {'yes' if e.batch else 'no'} "
                f"| {e.usage.input_tokens} | {e.usage.output_tokens} "
                f"| {e.usage.cache_creation_input_tokens} | {e.usage.cache_read_input_tokens} "
                f"| ${e.cost_usd:.4f} |"
            )
        summary = self.summary()
        lines += [
            "",
            f"**Total: ${summary.total_cost_usd:.4f}**"
            + (f" (cap ${self.cost_cap_usd:.2f})" if self.cost_cap_usd is not None else ""),
        ]
        if len(summary.by_iteration) > 1:
            per_iter = ", ".join(
                f"iteration {i}: ${cost:.4f}" for i, cost in sorted(summary.by_iteration.items())
            )
            lines.append(f"\nPer iteration: {per_iter}")
        return "\n".join(lines)

    def dump(self) -> list[dict[str, object]]:
        """JSON-safe entries for persistence."""
        return [entry.model_dump(mode="json") for entry in self._entries]

    @classmethod
    def load(cls, entries: list[dict[str, object]], cost_cap_usd: float | None = None) -> Ledger:
        """Rehydrate from persisted entries (used by resume). The cap is not
        re-enforced during load; the next record() call enforces it."""
        ledger = cls(cost_cap_usd=cost_cap_usd)
        ledger._entries = [LedgerEntry.model_validate(entry) for entry in entries]
        return ledger
