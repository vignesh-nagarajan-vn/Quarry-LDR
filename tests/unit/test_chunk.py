"""chunk_document: heading-bounded, overlapping, token-aware chunking.

The critical invariant under test throughout is that every chunk's
start_char/end_char slice doc.text back out exactly -- that is what lets
downstream citations point at exact spans.
"""

from __future__ import annotations

from itertools import pairwise

from quarry_ldr.config import ChunkSettings
from quarry_ldr.ingest.chunk import (
    Chunk,
    HeuristicTokenCounter,
    TokenCounter,
    chunk_document,
    make_chunk_id,
)
from quarry_ldr.ingest.extract import Block, ExtractedDoc


class WordCountCounter:
    """Counts whitespace-separated words instead of chars/4."""

    def count(self, text: str) -> int:
        return max(1, len(text.split()))


class FixedLargeCounter:
    """Always reports a huge token count, regardless of text length."""

    def count(self, text: str) -> int:
        return 10_000


def _sentences(n: int, prefix: str = "Sentence") -> str:
    """n short, roughly-uniform sentences, space separated."""
    return " ".join(f"{prefix} number {i} is right here." for i in range(n))


def test_empty_doc_returns_no_chunks() -> None:
    doc = ExtractedDoc(url="https://a.example", title="Empty", blocks=[])
    assert chunk_document(doc, ChunkSettings()) == []


def test_whitespace_only_blocks_return_no_chunks() -> None:
    doc = ExtractedDoc(
        url="https://a.example",
        title="Blank",
        blocks=[
            Block(heading_path=["Intro"], text="   \n  "),
            Block(heading_path=["Intro"], text=""),
        ],
    )
    assert chunk_document(doc, ChunkSettings()) == []


def test_slice_invariant_holds_for_every_chunk_multi_group() -> None:
    doc = ExtractedDoc(
        url="https://a.example",
        title="Multi",
        blocks=[
            Block(heading_path=["Intro"], text=_sentences(6)),
            Block(heading_path=["Results"], text=_sentences(5, prefix="Result")),
            Block(heading_path=["Results", "Latency"], text=_sentences(4, prefix="Latency")),
        ],
    )
    settings = ChunkSettings(target_tokens=40, overlap_tokens=8, min_tokens=5)
    chunks = chunk_document(doc, settings)
    assert chunks  # sanity: we actually produced chunks
    for chunk in chunks:
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text


def test_heading_boundaries_never_crossed() -> None:
    doc = ExtractedDoc(
        url="https://a.example",
        title="Headed",
        blocks=[
            Block(heading_path=["Intro"], text=_sentences(6)),
            Block(heading_path=["Results"], text=_sentences(5, prefix="Result")),
        ],
    )
    settings = ChunkSettings(target_tokens=30, overlap_tokens=5, min_tokens=5)
    chunks = chunk_document(doc, settings)
    intro_end = len(doc.blocks[0].text)
    for chunk in chunks:
        if chunk.heading_path == ["Intro"]:
            assert chunk.end_char <= intro_end
        else:
            assert chunk.heading_path == ["Results"]
            assert chunk.start_char >= intro_end + 2  # past the "\n\n" separator


def test_position_sequence_is_0_to_n_minus_1() -> None:
    doc = ExtractedDoc(
        url="https://a.example",
        title="Seq",
        blocks=[
            Block(heading_path=["Intro"], text=_sentences(6)),
            Block(heading_path=["Results"], text=_sentences(5, prefix="Result")),
        ],
    )
    settings = ChunkSettings(target_tokens=25, overlap_tokens=5, min_tokens=5)
    chunks = chunk_document(doc, settings)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_deterministic_across_two_runs() -> None:
    doc = ExtractedDoc(
        url="https://a.example",
        title="Det",
        blocks=[Block(heading_path=["Intro"], text=_sentences(10))],
    )
    settings = ChunkSettings(target_tokens=25, overlap_tokens=5, min_tokens=5)
    first = chunk_document(doc, settings)
    second = chunk_document(doc, settings)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first == second
    for c in first:
        assert c.chunk_id == make_chunk_id(doc.url, c.position)


def test_chunks_respect_target_tokens_within_one_sentence_of_slack() -> None:
    # No overlap so the packing bound is easy to state precisely: every
    # chunk's token count is <= target_tokens, unless it is a single
    # sentence that alone exceeds target_tokens (unavoidable slack).
    doc = ExtractedDoc(
        url="https://a.example",
        title="Bounded",
        blocks=[Block(heading_path=["Intro"], text=_sentences(20))],
    )
    settings = ChunkSettings(target_tokens=20, overlap_tokens=0, min_tokens=1)
    counter = HeuristicTokenCounter()
    chunks = chunk_document(doc, settings, counter)
    assert len(chunks) > 1
    for chunk in chunks:
        sentences = chunk.text.split(". ")
        if len(sentences) > 1:
            assert chunk.token_count <= settings.target_tokens


def test_consecutive_same_group_chunks_overlap_char_ranges() -> None:
    doc = ExtractedDoc(
        url="https://a.example",
        title="Overlap",
        blocks=[Block(heading_path=["Intro"], text=_sentences(20))],
    )
    settings = ChunkSettings(target_tokens=20, overlap_tokens=8, min_tokens=1)
    chunks = chunk_document(doc, settings)
    assert len(chunks) > 1
    for prev, cur in pairwise(chunks):
        assert prev.heading_path == cur.heading_path
        # Char ranges overlap: the new chunk starts before the previous ends.
        assert cur.start_char < prev.end_char
        # And the overlapping region is exact text agreement (shared suffix
        # of prev / shared prefix of cur), not merely overlapping ranges.
        overlap_len = prev.end_char - cur.start_char
        assert prev.text[-overlap_len:] == cur.text[:overlap_len]
        # Never the *entire* previous chunk repeated.
        assert cur.start_char > prev.start_char


def test_min_tokens_merges_trailing_fragment() -> None:
    # Craft a block whose last "sentence" is a short trailing fragment that,
    # packed alone, would fall under min_tokens.
    text = _sentences(8) + " Hi."
    doc = ExtractedDoc(
        url="https://a.example",
        title="Fragment",
        blocks=[Block(heading_path=["Intro"], text=text)],
    )
    # target_tokens=8 packs one ~33-char sentence per chunk (two would be 16
    # tokens, over budget), so the trailing "Hi." lands alone as its own
    # under-min_tokens fragment before the merge step folds it backward.
    settings = ChunkSettings(target_tokens=8, overlap_tokens=0, min_tokens=6)
    chunks = chunk_document(doc, settings)
    assert chunks
    # No chunk (other than possibly a lone chunk for a whole tiny group)
    # should end up under min_tokens once merging has run, since the
    # fragment always had a predecessor to merge into here.
    for chunk in chunks:
        assert chunk.token_count >= settings.min_tokens
    # The merge should have actually happened: the trailing fragment's own
    # short text ("Hi.") should not appear as a standalone final chunk.
    assert chunks[-1].text.strip().endswith("Hi.")
    assert chunks[-1].token_count >= settings.min_tokens


def test_single_chunk_group_stands_alone_even_under_min_tokens() -> None:
    doc = ExtractedDoc(
        url="https://a.example",
        title="Tiny",
        blocks=[Block(heading_path=["Intro"], text="Hi.")],
    )
    settings = ChunkSettings(target_tokens=512, overlap_tokens=64, min_tokens=48)
    chunks = chunk_document(doc, settings)
    assert len(chunks) == 1
    assert chunks[0].text == "Hi."
    assert chunks[0].token_count < settings.min_tokens


def test_doc_title_url_and_heading_path_filled_from_doc() -> None:
    doc = ExtractedDoc(
        url="https://example.test/article",
        title="My Article",
        blocks=[Block(heading_path=["Section", "Sub"], text=_sentences(3))],
    )
    chunks = chunk_document(doc, ChunkSettings())
    assert chunks
    for chunk in chunks:
        assert chunk.url == "https://example.test/article"
        assert chunk.doc_title == "My Article"
        assert chunk.heading_path == ["Section", "Sub"]


def test_custom_token_counter_changes_packing() -> None:
    doc = ExtractedDoc(
        url="https://a.example",
        title="Counter",
        blocks=[Block(heading_path=["Intro"], text=_sentences(10))],
    )
    settings = ChunkSettings(target_tokens=20, overlap_tokens=0, min_tokens=1)
    default_chunks = chunk_document(doc, settings, HeuristicTokenCounter())
    fixed_chunks = chunk_document(doc, settings, FixedLargeCounter())
    # FixedLargeCounter reports every candidate as far over target_tokens,
    # so each chunk can only ever hold a single mandatory sentence: one
    # chunk per sentence, unlike the default counter's multi-sentence packs.
    assert len(fixed_chunks) == 10
    assert len(default_chunks) != len(fixed_chunks)
    assert all(c.token_count == 10_000 for c in fixed_chunks)


def test_word_count_counter_is_honored_for_token_count_field() -> None:
    doc = ExtractedDoc(
        url="https://a.example",
        title="Words",
        blocks=[Block(heading_path=["Intro"], text="One two three four five.")],
    )
    chunks = chunk_document(doc, ChunkSettings(), WordCountCounter())
    assert len(chunks) == 1
    assert chunks[0].token_count == 5


def test_default_counter_used_when_none_passed() -> None:
    doc = ExtractedDoc(
        url="https://a.example",
        title="Default",
        blocks=[Block(heading_path=["Intro"], text="One two three four five.")],
    )
    chunks = chunk_document(doc, ChunkSettings(), None)
    heuristic = HeuristicTokenCounter()
    assert chunks[0].token_count == heuristic.count(chunks[0].text)


def test_no_chunk_crosses_a_heading_boundary_three_groups() -> None:
    doc = ExtractedDoc(
        url="https://a.example",
        title="Three",
        blocks=[
            Block(heading_path=["A"], text=_sentences(4, prefix="A")),
            Block(heading_path=["B"], text=_sentences(4, prefix="B")),
            Block(heading_path=["A"], text=_sentences(4, prefix="A2")),
        ],
    )
    settings = ChunkSettings(target_tokens=200, overlap_tokens=10, min_tokens=1)
    chunks = chunk_document(doc, settings)
    # heading_path "A" appears twice (non-adjacent groups are not merged).
    a_groups = [c.heading_path for c in chunks]
    assert a_groups.count(["A"]) == 2
    for chunk in chunks:
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text


def test_chunk_type_is_chunk_model() -> None:
    doc = ExtractedDoc(
        url="https://a.example",
        title="Type",
        blocks=[Block(heading_path=[], text="A single short sentence.")],
    )
    chunks = chunk_document(doc, ChunkSettings())
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)


def test_token_counter_protocol_accepts_any_conforming_object() -> None:
    counter: TokenCounter = WordCountCounter()
    assert counter.count("a b c") == 3
