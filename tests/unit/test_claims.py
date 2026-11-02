"""The shared reading of report prose: sentences, facts, figures, names."""

from __future__ import annotations

import pytest

from quarry_ldr.report.claims import (
    count_specifics,
    extract_figures,
    is_factual,
    is_gap_note,
    named_entities,
    split_sentences,
    strip_markers,
)


def test_split_keeps_markers_with_their_sentence() -> None:
    assert split_sentences("First claim here [1]. Second claim [2][3]. Third.") == [
        "First claim here [1].",
        "Second claim [2][3].",
        "Third.",
    ]


def test_split_handles_a_marker_after_the_period() -> None:
    assert split_sentences("First claim. [1] Second claim.") == [
        "First claim. [1]",
        "Second claim.",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Costs fell in the U.S. Department of Energy study [1].",
        "Some chemistries, e.g. Iron-air, are cheaper [1].",
        "Smith et al. Reported a gain of 3 percent [1].",
        "Leaders include Contemporary Amperex Technology Co. Limited (CATL) and Altris AB [1].",
    ],
)
def test_abbreviations_do_not_split(text: str) -> None:
    assert split_sentences(text) == [text]


def test_lowercase_continuation_does_not_split() -> None:
    assert len(split_sentences("Reached 1.5 MW. and then stopped working.")) == 1


@pytest.mark.parametrize(
    "sentence",
    [
        "Form Energy opened a plant in West Virginia.",
        "Round-trip efficiency reached 42 percent last year.",
        "These materials contribute to the overall performance and sustainability of the system.",
    ],
)
def test_factual_sentences(sentence: str) -> None:
    assert is_factual(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "Several challenges remain.",
        "What are the main challenges in this field today?",
        "This section reviews the deployment record across all the major markets of the world.",
        "The sources reviewed do not report measured flow rates from the recent demonstrations.",
        "No data on water use was found, and the evidence is missing for land use overall.",
        "Long-term degradation remains unclear across the published field trials to date.",
    ],
)
def test_non_factual_sentences(sentence: str) -> None:
    assert not is_factual(sentence)


def test_gap_note_detection() -> None:
    assert is_gap_note("The evidence does not quantify drilling cost trends.")
    assert not is_gap_note("Drilling costs fell 30 percent according to the evidence.")


def test_figures_normalize_and_skip_markers() -> None:
    assert extract_figures("1,200 Wh/kg and 46.250 percent in 2026 [3]") == {
        "1200",
        "46.25",
        "2026",
    }
    assert extract_figures("Fe²⁺ with no digits") == set()
    assert extract_figures("a range of 30-40%") == {"30", "40"}


def test_named_entities_drop_common_openers() -> None:
    names = named_entities(
        "The company Form Energy built it. In Minnesota, Xcel Energy agreed [1]."
    )
    assert {"Form Energy", "Minnesota", "Xcel Energy"} <= names
    assert "The" not in names and "In" not in names


def test_count_specifics_counts_names_and_figures() -> None:
    assert count_specifics("Form Energy targets 100 hours.") == 2
    assert count_specifics("it is cheap and abundant") == 0


def test_strip_markers_tidies_punctuation() -> None:
    assert strip_markers("A claim [1][2], then more [3].") == "A claim, then more."
