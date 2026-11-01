"""Deterministic report cleanup: every rule, offline, no model."""

from __future__ import annotations

from quarry_ldr.ingest.chunk import Chunk
from quarry_ldr.pipeline.synthesize import DraftReport, ReportSection
from quarry_ldr.report.citations import CitationIndex
from quarry_ldr.report.cleanup import (
    clean_draft,
    drop_repeats,
    fix_markers,
    normalize_markers,
    split_walls,
    strip_conflict_denials,
    strip_question_leaks,
    strip_title_headings,
)


def _index(n: int) -> CitationIndex:
    index = CitationIndex()
    for i in range(n):
        index.add(
            Chunk(
                chunk_id=f"c{i}",
                url=f"https://s{i}.example/a",
                text="t",
                token_count=1,
                position=i,
                start_char=0,
                end_char=1,
            )
        )
    return index


def test_compound_markers_become_plain_runs() -> None:
    assert normalize_markers("Costs fell [2 (sq02)].") == ("Costs fell [2].", 1)
    assert normalize_markers("Costs fell [1, 2].")[0] == "Costs fell [1][2]."
    assert normalize_markers("Costs fell [1; 3].")[0] == "Costs fell [1][3]."
    assert normalize_markers("Costs fell [1-3].")[0] == "Costs fell [1][2][3]."
    assert normalize_markers("Costs fell [4\u20135].")[0] == "Costs fell [4][5]."


def test_plain_markers_and_links_are_untouched() -> None:
    text = "See [the study](https://a.example) and [3]."
    assert normalize_markers(text) == (text, 0)


def test_wide_or_backwards_ranges_are_removed_whole() -> None:
    assert normalize_markers("Many say so [1-40].") == ("Many say so .", 1)
    assert fix_markers("Many say so [1-40].", _index(1))[0] == "Many say so."
    assert normalize_markers("Many say so [5-2].")[0] == "Many say so ."


def test_unresolved_markers_are_stripped_after_normalizing() -> None:
    text, n_normalized, n_stripped = fix_markers("A [1, 9] and B [2 (sq01)].", _index(2))
    assert text == "A [1] and B [2]."
    assert (n_normalized, n_stripped) == (2, 1)


def test_leaked_ids_and_restated_questions_are_removed() -> None:
    questions = ["What do iron-air batteries cost?"]
    text = (
        "What do iron-air batteries cost? Costs reached $20 per kWh [1].\n"
        "sq01: Costs vary by site [2].\n"
        "Plants follow (sq01) the grid [3] [sq02]."
    )
    cleaned, count = strip_question_leaks(text, questions)
    assert cleaned.splitlines() == [
        "Costs reached $20 per kWh [1].",
        "Costs vary by site [2].",
        "Plants follow the grid [3].",
    ]
    assert count == 4


def test_a_line_that_is_only_the_question_disappears() -> None:
    cleaned, _ = strip_question_leaks("Intro.\nWhat do they cost?\nBody.", ["What do they cost?"])
    assert cleaned == "Intro.\nBody."


def test_repeated_sentences_are_dropped_across_sections() -> None:
    sentence = "Form Energy plans a 1.5 MW pilot in Minnesota by 2025"
    sections = [
        ReportSection(title="A", markdown=f"{sentence} [1]. Short one."),
        ReportSection(title="B", markdown=f"New fact here [2]. {sentence} [3]. Short one."),
    ]
    cleaned, count = drop_repeats(sections)
    assert count == 1
    assert cleaned[1].markdown == "New fact here [2]. Short one."


def test_wall_paragraphs_split_at_sentence_boundaries() -> None:
    sentence = "Iron-air cells store energy for one hundred hours at low cost [1]."
    wall = " ".join([sentence] * 30)  # 360 words
    split, count = split_walls(f"Lead.\n\n{wall}", max_words=160)
    paragraphs = split.split("\n\n")
    assert count == 1
    assert len(paragraphs) == 1 + 3
    assert all(len(p.split()) <= 160 for p in paragraphs)
    assert " ".join(paragraphs[1:]) == wall  # nothing lost or reordered


def test_lists_and_tables_are_never_split() -> None:
    items = "\n".join(f"- item {i} " + "word " * 20 for i in range(12))
    assert split_walls(items, max_words=50) == (items, 0)


def test_clean_draft_runs_every_rule_and_counts() -> None:
    sentence = "Halvard Grid commissioned a 5 MW plant in Tromso in 2027"
    draft = DraftReport(
        topic="t",
        sections=[
            ReportSection(title="One", markdown=f"{sentence} [1, 2]. Is it cheap? sq01 [7]."),
            ReportSection(title="Two", markdown=f"{sentence} [1]."),
        ],
    )
    cleaned, counts = clean_draft(draft, _index(2), ["Is it cheap?"], 160)
    assert cleaned.sections[0].markdown == f"{sentence} [1][2]."
    assert [s.title for s in cleaned.sections] == ["One"]  # "Two" emptied, so removed
    assert counts.markers_normalized == 1
    assert counts.markers_stripped == 1
    assert counts.question_leaks_removed == 2
    assert counts.repeats_dropped == 1


def test_subheadings_repeating_the_section_title_are_removed() -> None:
    text = "### Primary challenges\n\nBody text.\n\n### Different heading\n\nMore."
    assert strip_title_headings(text, "Primary Challenges") == (
        "\nBody text.\n\n### Different heading\n\nMore.",
        1,
    )


def test_uncited_conflict_denials_are_removed_and_real_text_stays() -> None:
    denial = "The sources reviewed do not provide conflicting evidence on these factors."
    text = f"Costs fell [1]. {denial} Plants grew [2]."
    assert strip_conflict_denials(text) == ("Costs fell [1]. Plants grew [2].", 1)


def test_conflict_rule_leaves_other_sentences_alone() -> None:
    kept = [
        "Studies do not contradict the 2024 figure [3].",  # cited, so the evidence decides
        "There was no conflict of interest at the plant.",  # not about the sources
        "- The sources reviewed show no disagreement.",  # lists are not prose
    ]
    for text in kept:
        assert strip_conflict_denials(text) == (text, 0)


def test_clean_draft_counts_the_new_rules() -> None:
    draft = DraftReport(
        topic="t",
        sections=[
            ReportSection(
                title="Costs",
                markdown="### Costs\n\nCosts fell [1]. The evidence shows no conflicting data.",
            )
        ],
    )
    cleaned, counts = clean_draft(draft, _index(1), [], 160)
    assert cleaned.sections[0].markdown == "Costs fell [1]."
    assert counts.title_headings_removed == 1
    assert counts.conflict_denials_removed == 1


def test_tables_and_rules_survive_leak_stripping() -> None:
    table = "| Plant | MW |\n|---|---|\n| Tromso | 5 [1] |\n\n---"
    assert strip_question_leaks(table, ["What is it?"]) == (table, 0)
