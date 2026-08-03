"""Unit tests for extract_document over the synthetic fixture corpus.

trafilatura is pure parsing (no network), so these run straight against the
fixture HTML on disk. Fixture URLs come from manifest.json rather than being
hardcoded, matching the rest of the suite's convention (see test_fixtures.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import quarry_ldr.ingest.extract as extract_mod
from quarry_ldr.ingest.extract import Block, ExtractedDoc, extract_document


def _manifest(fixtures_dir: Path) -> list[dict[str, object]]:
    data = json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))
    return list(data["docs"])


def _entry(fixtures_dir: Path, filename: str) -> dict[str, object]:
    for doc in _manifest(fixtures_dir):
        if doc["file"] == filename:
            return doc
    raise AssertionError(f"{filename} not found in manifest.json")


def _html_bytes(fixtures_dir: Path, filename: str) -> bytes:
    return (fixtures_dir / "html" / filename).read_bytes()


def test_normal_doc_extracts_title_and_multiple_blocks(fixtures_dir: Path) -> None:
    entry = _entry(fixtures_dir, "f09.html")
    html = _html_bytes(fixtures_dir, "f09.html")

    doc = extract_document(html, str(entry["url"]))

    assert doc is not None
    assert doc.url == entry["url"]
    assert doc.title == entry["title"]
    assert len(doc.blocks) >= 2
    assert all(isinstance(block, Block) for block in doc.blocks)


def test_normal_doc_text_contains_known_sentence_fragment(fixtures_dir: Path) -> None:
    entry = _entry(fixtures_dir, "f09.html")
    html = _html_bytes(fixtures_dir, "f09.html")

    doc = extract_document(html, str(entry["url"]))

    assert doc is not None
    assert "quartz sand" in doc.text


def test_heading_path_values_are_drawn_from_the_docs_h2_titles(fixtures_dir: Path) -> None:
    entry = _entry(fixtures_dir, "f09.html")
    html = _html_bytes(fixtures_dir, "f09.html")
    expected_headings = {
        "Costs and financing",
        "Grid integration",
        "Open questions",
        "Community and siting",
        "How the pilot works",
    }

    doc = extract_document(html, str(entry["url"]))

    assert doc is not None
    seen_headings = {block.heading_path[-1] for block in doc.blocks if block.heading_path}
    assert seen_headings == expected_headings
    # Every block sits under exactly one of those section headings.
    assert all(len(block.heading_path) == 1 for block in doc.blocks)


def test_heading_path_never_contains_the_doc_title(fixtures_dir: Path) -> None:
    entry = _entry(fixtures_dir, "f09.html")
    html = _html_bytes(fixtures_dir, "f09.html")

    doc = extract_document(html, str(entry["url"]))

    assert doc is not None
    for block in doc.blocks:
        assert doc.title not in block.heading_path


def test_paywall_fixture_returns_none(fixtures_dir: Path) -> None:
    entry = _entry(fixtures_dir, "f37.html")
    html = _html_bytes(fixtures_dir, "f37.html")

    assert extract_document(html, str(entry["url"])) is None


def test_js_only_fixture_returns_none(fixtures_dir: Path) -> None:
    entry = _entry(fixtures_dir, "f38.html")
    html = _html_bytes(fixtures_dir, "f38.html")

    assert extract_document(html, str(entry["url"])) is None


def test_bad_encoding_fixture_does_not_raise(fixtures_dir: Path) -> None:
    entry = _entry(fixtures_dir, "f39.html")
    html = _html_bytes(fixtures_dir, "f39.html")

    # Sanity check this really is malformed utf-8, matching test_fixtures.py.
    with pytest.raises(UnicodeDecodeError):
        html.decode("utf-8")

    doc = extract_document(html, str(entry["url"]))

    assert doc is None or "Temperatur" in doc.text


def test_str_and_bytes_inputs_agree_on_a_normal_doc(fixtures_dir: Path) -> None:
    entry = _entry(fixtures_dir, "f01.html")
    raw = _html_bytes(fixtures_dir, "f01.html")
    as_str = raw.decode("utf-8")

    doc_from_bytes = extract_document(raw, str(entry["url"]))
    doc_from_str = extract_document(as_str, str(entry["url"]))

    assert doc_from_bytes is not None
    assert doc_from_bytes == doc_from_str


def test_blocks_are_nonempty_strings_across_several_normal_docs(fixtures_dir: Path) -> None:
    normal_docs = [d for d in _manifest(fixtures_dir) if d["kind"] == "normal"][:5]
    assert normal_docs

    for entry in normal_docs:
        html = _html_bytes(fixtures_dir, str(entry["file"]))
        doc = extract_document(html, str(entry["url"]))
        assert doc is not None, f"{entry['file']} unexpectedly returned None"
        for block in doc.blocks:
            assert isinstance(block.text, str)
            assert block.text.strip() != ""


def test_extracted_doc_text_joins_blocks_with_blank_lines() -> None:
    doc = ExtractedDoc(
        url="https://joined.example/x",
        title="T",
        blocks=[Block(text="first"), Block(text="second"), Block(text="third")],
    )

    assert doc.text == "first\n\nsecond\n\nthird"


def test_tiny_block_merges_into_previous_block() -> None:
    html = (
        "<html><head><title>Merge Test</title></head><body><article>"
        "<h1>Merge Test</h1>"
        "<h2>Section One</h2>"
        "<p>This is the first paragraph in section one and it is definitely long "
        "enough to survive extraction easily on its own as a full standalone "
        "block of text here.</p>"
        "<p>Tiny note.</p>"
        "<p>This is the second full paragraph in section one, also long enough "
        "by itself to survive as its own block without any merging needed at "
        "all today.</p>"
        "</article></body></html>"
    )

    doc = extract_document(html, "https://merge.example/test")

    assert doc is not None
    # The 10-char "Tiny note." block (<40 chars) folds into the block before
    # it rather than standing alone.
    assert len(doc.blocks) == 2
    assert doc.blocks[0].text.endswith("Tiny note.")
    assert "Tiny note." not in doc.blocks[1].text


def test_published_date_extracted_as_string(fixtures_dir: Path) -> None:
    entry = _entry(fixtures_dir, "f01.html")
    html = _html_bytes(fixtures_dir, "f01.html")

    doc = extract_document(html, str(entry["url"]))

    assert doc is not None
    assert doc.published == "2024-12-08"
    assert isinstance(doc.published, str)


def test_title_falls_back_to_title_tag_when_metadata_title_is_empty(
    monkeypatch: pytest.MonkeyPatch, fixtures_dir: Path
) -> None:
    entry = _entry(fixtures_dir, "f09.html")
    html = _html_bytes(fixtures_dir, "f09.html")

    class _EmptyMeta:
        title = ""
        language: str | None = None
        date: str | None = None

    monkeypatch.setattr(extract_mod.trafilatura, "extract_metadata", lambda *a, **k: _EmptyMeta())

    doc = extract_document(html, str(entry["url"]))

    assert doc is not None
    # trafilatura's own metadata title was forced empty; the fallback still
    # finds the literal <title> tag text.
    assert doc.title == str(entry["title"])


def test_robots_disallowed_fixture_still_extracts(fixtures_dir: Path) -> None:
    """Robots compliance is the fetcher's job (fetch.py), not extraction's:
    extract_document has no opinion on whether a URL was allowed to be fetched."""
    entry = _entry(fixtures_dir, "f40.html")
    html = _html_bytes(fixtures_dir, "f40.html")

    doc = extract_document(html, str(entry["url"]))

    assert doc is not None
    assert len(doc.blocks) >= 2
    assert "quartz sand" in doc.text
