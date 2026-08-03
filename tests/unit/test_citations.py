"""Citation numbering, resolution, and hard failure on dangling markers."""

from __future__ import annotations

import pytest

from quarry_ldr.ingest.chunk import Chunk
from quarry_ldr.report.citations import CitationError, CitationIndex


def make_chunk(chunk_id: str, url: str = "https://vesterholm-times.example/a") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        url=url,
        doc_title="Doc",
        heading_path=["Background"],
        text="text",
        token_count=10,
        position=0,
        start_char=100,
        end_char=200,
    )


def test_add_assigns_sequential_stable_numbers() -> None:
    index = CitationIndex()
    a = index.add(make_chunk("aaa"))
    b = index.add(make_chunk("bbb"))
    again = index.add(make_chunk("aaa"))
    assert (a, b) == (1, 2)
    assert again == 1
    assert len(index) == 2


def test_get_resolves_to_url_and_offsets() -> None:
    index = CitationIndex()
    index.add(make_chunk("aaa", url="https://polarforsk.example/notes"))
    citation = index.get(1)
    assert citation.url == "https://polarforsk.example/notes"
    assert (citation.start_char, citation.end_char) == (100, 200)
    assert citation.heading_path == ["Background"]


def test_unknown_number_raises() -> None:
    index = CitationIndex()
    with pytest.raises(CitationError, match=r"\[7\]"):
        index.get(7)


def test_validate_markdown_finds_dangling_marker() -> None:
    index = CitationIndex()
    index.add(make_chunk("aaa"))
    index.validate_markdown("A claim [1]. Another [1].")
    with pytest.raises(CitationError):
        index.validate_markdown("A claim [1] and a dangling one [2].")


def test_numbers_in_order() -> None:
    index = CitationIndex()
    assert index.numbers_in("x [3] y [1] z [12]") == [3, 1, 12]
    assert index.numbers_in("no markers") == []


def test_references_markdown_lists_all() -> None:
    index = CitationIndex()
    index.add(make_chunk("aaa", url="https://a.example/1"))
    index.add(make_chunk("bbb", url="https://b.example/2"))
    references = index.references_markdown()
    assert "## References" in references
    assert "[1]" in references and "https://a.example/1" in references
    assert "[2]" in references and "https://b.example/2" in references
    assert "chars 100-200" in references
