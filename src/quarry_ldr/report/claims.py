"""Deterministic reading of report prose: sentences, facts, figures, and names.

Shared by VERIFY (what must be checked), GAP (what evidence still lacks), and
the scorecard (what a finished report contains), so all three agree on what a
factual sentence is. No model calls.
"""

from __future__ import annotations

import re

from quarry_ldr.report.citations import CITATION_MARKER

_BOUNDARY = re.compile(r"[.!?](?:\s*\[\d+\])*\s+(?=[A-Z0-9(\[\"'“*])")
_ABBREVIATION = re.compile(
    r"(?:\b(?:e\.g|i\.e|vs|etc|al|Dr|Prof|Mr|Ms|Mrs|St|Fig|No|approx|cf|Inc|Ltd|Corp)|"
    r"\b(?:[A-Z]\.)+[A-Z]?)\.?$"
)
_FIGURE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\d])")
_ENTITY = re.compile(r"\b[A-Z][\w&'-]*(?:\s+(?:of\s+)?[A-Z][\w&'-]*)*")
_GAP_NOTE = re.compile(
    r"\b(?:evidence|sources?(?: reviewed)?|data|literature|information|studies)\b.{0,60}?"
    r"\b(?:missing|lack|lacks|absent|unavailable|insufficient|silent|limited|not (?:provide|"
    r"report|address|cover|include|available|establish|contain|quantify)|does not|do not|did not|"
    r"no (?:data|evidence|figures?|information))\b|"
    r"\b(?:could not be (?:established|confirmed|verified)|remains? (?:unclear|unknown|"
    r"unresolved)|not (?:publicly )?(?:reported|available|disclosed)|sources? (?:disagree|conflict))\b",
    re.IGNORECASE,
)
_META = re.compile(r"^(?:this|the) (?:section|report|chapter)\b", re.IGNORECASE)
_MIN_FACTUAL_WORDS = 6
_ALWAYS_FACTUAL_WORDS = 10
_COMMON_OPENERS = frozenset(
    {
        "A",
        "An",
        "And",
        "As",
        "At",
        "But",
        "By",
        "For",
        "From",
        "He",
        "How",
        "If",
        "In",
        "It",
        "Its",
        "Most",
        "No",
        "Of",
        "On",
        "One",
        "Or",
        "She",
        "So",
        "Some",
        "That",
        "The",
        "Their",
        "There",
        "These",
        "They",
        "This",
        "Those",
        "To",
        "We",
        "What",
        "When",
        "Where",
        "While",
        "Who",
        "With",
    }
)
# Capitalized units are figures' companions, not names ("1.5 MW" names nobody).
_UNITS = frozenset(
    {"Ah", "C", "EUR", "GBP", "GJ", "GW", "GWh", "J", "K", "MJ", "MPa", "MW", "MWh", "Pa"}
    | {"TW", "TWh", "USD", "V", "W", "Wh"}
)


def strip_markers(text: str) -> str:
    """The sentence without [n] markers, whitespace tidied."""
    return re.sub(
        r"\s+([.,;:!?])", r"\1", re.sub(r"\s+", " ", CITATION_MARKER.sub("", text))
    ).strip()


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping trailing markers with their sentence.

    Abbreviations and initials do not end a sentence: a false split would make
    half a sentence look uncited, and VERIFY removes uncited facts.
    """
    pieces: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        candidate = text[start : match.start() + 1]
        if _ABBREVIATION.search(candidate):
            continue
        pieces.append(text[start : match.end()].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def is_gap_note(sentence: str) -> bool:
    """A sentence that reports a limit of the evidence, not a fact about the topic."""
    return bool(_GAP_NOTE.search(sentence))


def is_factual(sentence: str) -> bool:
    """Whether an uncited sentence asserts something that needs a source.

    Short linking sentences, questions, section signposting, and labeled gap
    notes do not. A sentence with a figure, a mid-sentence name, or ten or
    more words does.
    """
    bare = strip_markers(sentence).lstrip("-*0123456789.) ").strip()
    words = bare.split()
    if len(words) < _MIN_FACTUAL_WORDS or bare.endswith("?"):
        return False
    if _META.match(bare) or is_gap_note(bare):
        return False
    if len(words) >= _ALWAYS_FACTUAL_WORDS or extract_figures(bare):
        return True
    return bool(_ENTITY.search(" ".join(words[1:])))


def extract_figures(text: str) -> set[str]:
    """Every number in ``text`` normalized for comparison ("1,200.0" is "1200").

    Citation markers are not figures.
    """
    figures: set[str] = set()
    for match in _FIGURE.finditer(CITATION_MARKER.sub(" ", text)):
        raw = match.group(0).replace(",", "").rstrip(".")
        if "." in raw:
            raw = raw.rstrip("0").rstrip(".")
        figures.add(raw.lstrip("0") or "0")
    return figures


def named_entities(text: str) -> set[str]:
    """Distinct capitalized names, with sentence-opening common words removed."""
    names: set[str] = set()
    for match in _ENTITY.finditer(CITATION_MARKER.sub(" ", text)):
        words = match.group(0).split()
        while words and words[0] in _COMMON_OPENERS:
            words.pop(0)
        words = [word for word in words if word not in _UNITS]
        name = " ".join(words)
        if len(name) >= 2:
            names.add(name)
    return names


def count_specifics(text: str) -> int:
    """Distinct names and figures: how concrete a passage is."""
    return len(named_entities(text)) + len(extract_figures(text))
