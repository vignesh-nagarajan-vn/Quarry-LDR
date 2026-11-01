"""Deterministic report cleanup: markers, leaked plan text, repeats, walls.

Every engine's draft passes through ``clean_draft`` before VERIFY, and the
marker rules run again at RENDER so a draft persisted by an older build (or
replayed on resume) can never fail on a marker the index does not know. No
model calls: these are the edits a model must not be trusted to make.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from quarry_ldr.pipeline.synthesize import DraftReport, ReportSection
from quarry_ldr.report.citations import CITATION_MARKER, CitationError, CitationIndex
from quarry_ldr.report.claims import split_sentences, strip_markers

# A bracket whose content starts with a number and holds only numbers,
# separators, ranges, and sub-question ids: "[2 (sq02)]", "[1, 2]", "[1-3]".
# Markdown links ("[text](url)") start with a letter and never match.
_COMPOUND_MARKER = re.compile(
    r"\[(\d+(?:\s*(?:[,;]|[-\u2013])\s*\d+)*(?:\s*[,;]?\s*\(?sq\d+\)?)*)\](?!\()"
)
_RANGE = re.compile(r"(\d+)\s*[-\u2013]\s*(\d+)")
_SQ_ID = re.compile(r"\s*\(?\bsq\d{2,}\b\)?:?", re.IGNORECASE)
_EMPTY_BRACKETS = re.compile(r"\s*\[\s*\]")
_GAP_BEFORE_PUNCT = re.compile(r"[ \t]+([.,;:!?])")
# The widest range expanded; "[1-40]" is not a citation anyone can check.
_MAX_RANGE = 6
# Shorter sentences repeat legitimately ("Costs vary.").
_MIN_REPEAT_WORDS = 6
_BLOCK_START = re.compile(r"^\s*(?:[-*+>|#]|\d+[.)]\s|```)")


class CleanupCounts(BaseModel):
    """What cleanup changed, logged and scored per run."""

    markers_normalized: int = 0
    markers_stripped: int = 0
    question_leaks_removed: int = 0
    repeats_dropped: int = 0
    paragraphs_split: int = 0


def normalize_markers(text: str) -> tuple[str, int]:
    """Rewrite compound markers as plain ``[n]`` runs.

    ``[2 (sq02)]`` becomes ``[2]``, ``[1, 2]`` becomes ``[1][2]``, and
    ``[1-3]`` becomes ``[1][2][3]``. A range wider than six, or backwards,
    is unverifiable and is removed whole.
    """
    count = 0

    def rewrite(match: re.Match[str]) -> str:
        nonlocal count
        body = _SQ_ID.sub("", match.group(1))
        numbers: list[int] = []
        for part in re.split(r"\s*[,;]\s*", body.strip()):
            if not part:
                continue
            span = _RANGE.fullmatch(part)
            if span:
                low, high = int(span.group(1)), int(span.group(2))
                if high < low or high - low >= _MAX_RANGE:
                    count += 1
                    return ""
                numbers.extend(range(low, high + 1))
            elif part.isdigit():
                numbers.append(int(part))
        rebuilt = "".join(f"[{n}]" for n in numbers)
        if rebuilt != match.group(0):
            count += 1
        return rebuilt

    return _COMPOUND_MARKER.sub(rewrite, text), count


def strip_unresolved(text: str, citations: CitationIndex) -> tuple[str, int]:
    """Remove ``[n]`` markers the index does not know, with their count."""
    count = 0

    def keep_or_drop(match: re.Match[str]) -> str:
        nonlocal count
        try:
            citations.get(int(match.group(1)))
        except CitationError:
            count += 1
            return ""
        return match.group(0)

    return CITATION_MARKER.sub(keep_or_drop, text), count


def fix_markers(text: str, citations: CitationIndex) -> tuple[str, int, int]:
    """Normalize, then strip what still does not resolve."""
    normalized, n_normalized = normalize_markers(text)
    stripped, n_stripped = strip_unresolved(normalized, citations)
    if n_stripped or n_normalized:
        # "on [999]." left "on ." behind; close the gap a removed marker opened.
        stripped = _GAP_BEFORE_PUNCT.sub(r"\1", stripped)
    return stripped, n_normalized, n_stripped


def _norm(sentence: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", strip_markers(sentence).lower()).strip()


def strip_question_leaks(text: str, questions: list[str]) -> tuple[str, int]:
    """Remove sub-question ids and sentences that restate a plan question.

    Writers were observed echoing "sq02:" labels and pasting the question as
    a sentence; neither is part of an answer.
    """
    asked = {_norm(question) for question in questions}
    count = 0
    lines: list[str] = []
    for line in text.split("\n"):
        without_ids, n_ids = _SQ_ID.subn("", line)
        if n_ids:
            without_ids = _EMPTY_BRACKETS.sub("", without_ids)
        count += n_ids
        if not line[:1].isspace():
            without_ids = without_ids.lstrip()
        sentences = split_sentences(without_ids)
        # A sentence emptied by the id removal ("sq01 .") goes with them.
        kept = [s for s in sentences if (_norm(s) or not n_ids) and _norm(s) not in asked]
        count += sum(1 for s in sentences if _norm(s) in asked)
        if sentences and not kept:
            continue
        indent = without_ids[: len(without_ids) - len(without_ids.lstrip())]
        lines.append(indent + " ".join(kept) if len(kept) != len(sentences) else without_ids)
    return "\n".join(lines), count


def drop_repeats(sections: list[ReportSection]) -> tuple[list[ReportSection], int]:
    """Drop any sentence of six or more words already said earlier in the report."""
    seen: set[str] = set()
    count = 0
    cleaned: list[ReportSection] = []
    for section in sections:
        lines: list[str] = []
        for line in section.markdown.split("\n"):
            sentences = split_sentences(line)
            kept: list[str] = []
            for sentence in sentences:
                key = _norm(sentence)
                if len(key.split()) >= _MIN_REPEAT_WORDS and key in seen:
                    count += 1
                    continue
                seen.add(key)
                kept.append(sentence)
            if sentences and not kept:
                continue
            if len(kept) == len(sentences):
                lines.append(line)
            else:
                indent = line[: len(line) - len(line.lstrip())]
                lines.append(indent + " ".join(kept))
        cleaned.append(section.model_copy(update={"markdown": "\n".join(lines)}))
    return cleaned, count


def split_walls(text: str, max_words: int) -> tuple[str, int]:
    """Split prose paragraphs over ``max_words`` at sentence boundaries.

    Lists, tables, quotes, headings, and code are left alone.
    """
    count = 0
    out: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        words = len(paragraph.split())
        if words <= max_words or _BLOCK_START.match(paragraph) or "\n" in paragraph.strip():
            out.append(paragraph)
            continue
        sentences = split_sentences(paragraph.strip())
        n_parts = -(-words // max_words)
        target = words / n_parts
        parts: list[list[str]] = [[]]
        filled = 0
        for sentence in sentences:
            if parts[-1] and filled + len(sentence.split()) > target and len(parts) < n_parts:
                parts.append([])
                filled = 0
            parts[-1].append(sentence)
            filled += len(sentence.split())
        if len(parts) > 1:
            count += 1
        out.append("\n\n".join(" ".join(part) for part in parts))
    return "\n\n".join(out), count


def clean_draft(
    draft: DraftReport,
    citations: CitationIndex,
    questions: list[str],
    max_paragraph_words: int,
) -> tuple[DraftReport, CleanupCounts]:
    """Every deterministic rule, in order, over every section."""
    counts = CleanupCounts()
    sections: list[ReportSection] = []
    for section in draft.sections:
        text, n_normalized, n_stripped = fix_markers(section.markdown, citations)
        text, n_leaks = strip_question_leaks(text, questions)
        counts.markers_normalized += n_normalized
        counts.markers_stripped += n_stripped
        counts.question_leaks_removed += n_leaks
        sections.append(section.model_copy(update={"markdown": text}))
    sections, counts.repeats_dropped = drop_repeats(sections)
    split: list[ReportSection] = []
    for section in sections:
        text, n_split = split_walls(section.markdown, max_paragraph_words)
        counts.paragraphs_split += n_split
        if text.strip():  # a section emptied by cleanup is removed, not rendered bare
            split.append(section.model_copy(update={"markdown": text.strip()}))
    return draft.model_copy(update={"sections": split}), counts
