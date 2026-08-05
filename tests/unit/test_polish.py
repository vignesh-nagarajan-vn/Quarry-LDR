"""polish_draft: the citation-multiset and delimiter round-trip guards."""

from __future__ import annotations

from typing import Any, cast

import pytest

from quarry_ldr.config import QuarryConfig
from quarry_ldr.ledger import CostCapExceeded, TokenUsage
from quarry_ldr.pipeline.synthesize import DraftReport, ReportSection, polish_draft
from quarry_ldr.providers.base import CompletionResult, Provider


def make_draft() -> DraftReport:
    return DraftReport(
        topic="t",
        sections=[
            ReportSection(title="Overview", markdown="Rough opening [1]. It cites twice [1]."),
            ReportSection(title="Details", markdown="Detail claim [2]."),
        ],
    )


class ScriptedPolishProvider:
    """Returns a fixed response text, or echoes the prompt (identity polish)."""

    def __init__(self, response: str | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> CompletionResult:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        text = self.response if self.response is not None else kwargs["prompt"]
        return CompletionResult(
            text=text,
            usage=TokenUsage(input_tokens=10, output_tokens=10),
            model=kwargs["model"],
            stop_reason="end_turn",
        )


async def test_identity_polish_round_trips(cfg: QuarryConfig) -> None:
    provider = ScriptedPolishProvider()
    draft = make_draft()
    polished = await polish_draft(draft, cast(Provider, provider), cfg)
    assert [s.title for s in polished.sections] == ["Overview", "Details"]
    assert polished.sections[0].markdown == "Rough opening [1]. It cites twice [1]."
    call = provider.calls[0]
    assert call["model"] == cfg.models.assisted
    assert call["stage"] == "polish"
    assert "<!--Q-SECTION:Overview-->" in call["prompt"]


async def test_improved_prose_with_preserved_markers_is_accepted(cfg: QuarryConfig) -> None:
    response = (
        "<!--Q-SECTION:Overview-->\n\n"
        "A polished opening [1], citing twice [1].\n\n"
        "<!--Q-SECTION:Details-->\n\n"
        "A polished detail claim [2]."
    )
    polished = await polish_draft(
        make_draft(), cast(Provider, ScriptedPolishProvider(response)), cfg
    )
    assert polished.sections[0].markdown == "A polished opening [1], citing twice [1]."
    assert polished.sections[1].markdown == "A polished detail claim [2]."


async def test_changed_marker_multiset_discards(cfg: QuarryConfig) -> None:
    # One [1] lost: multiset {1: 2, 2: 1} became {1: 1, 2: 1}.
    response = (
        "<!--Q-SECTION:Overview-->\n\nA polished opening [1].\n\n"
        "<!--Q-SECTION:Details-->\n\nA polished detail claim [2]."
    )
    draft = make_draft()
    polished = await polish_draft(draft, cast(Provider, ScriptedPolishProvider(response)), cfg)
    assert polished is draft  # the local draft stands


async def test_preamble_before_first_delimiter_discards(cfg: QuarryConfig) -> None:
    response = (
        "Here is your polished report:\n\n"
        "<!--Q-SECTION:Overview-->\n\nRough opening [1]. It cites twice [1].\n\n"
        "<!--Q-SECTION:Details-->\n\nDetail claim [2]."
    )
    draft = make_draft()
    polished = await polish_draft(draft, cast(Provider, ScriptedPolishProvider(response)), cfg)
    assert polished is draft


async def test_retitled_sections_discard(cfg: QuarryConfig) -> None:
    response = (
        "<!--Q-SECTION:Introduction-->\n\nRough opening [1]. It cites twice [1].\n\n"
        "<!--Q-SECTION:Details-->\n\nDetail claim [2]."
    )
    draft = make_draft()
    polished = await polish_draft(draft, cast(Provider, ScriptedPolishProvider(response)), cfg)
    assert polished is draft


async def test_api_failure_is_fail_soft(cfg: QuarryConfig) -> None:
    provider = ScriptedPolishProvider(error=RuntimeError("api down"))
    draft = make_draft()
    polished = await polish_draft(draft, cast(Provider, provider), cfg)
    assert polished is draft


async def test_cost_cap_always_propagates(cfg: QuarryConfig) -> None:
    provider = ScriptedPolishProvider(error=CostCapExceeded(1.0, 0.5))
    with pytest.raises(CostCapExceeded):
        await polish_draft(make_draft(), cast(Provider, provider), cfg)
