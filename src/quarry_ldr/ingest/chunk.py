"""Token-aware chunking: ~512 tokens, 64 overlap, heading path preserved.

Token counting is pluggable via the TokenCounter protocol: runtime uses the
embedder's HF tokenizer, tests use the deterministic offline heuristic so the
suite needs no model downloads and no network.

Implemented in M4.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

from pydantic import BaseModel, Field

from quarry_ldr.config import ChunkSettings
from quarry_ldr.ingest.extract import ExtractedDoc


class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        """Number of tokens in ``text`` under this counter's tokenization."""
        ...


class HeuristicTokenCounter:
    """Deterministic offline approximation: ~4 characters per token."""

    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


class Chunk(BaseModel):
    """The atomic unit of evidence. chunk_id is deterministic, so re-chunking
    the same document yields the same ids and the index add is idempotent."""

    chunk_id: str
    url: str
    doc_title: str = ""
    heading_path: list[str] = Field(default_factory=list)
    text: str
    token_count: int
    position: int
    start_char: int
    end_char: int


def make_chunk_id(url: str, position: int) -> str:
    """Deterministic id: sha256(url|position), 16 hex chars."""
    digest = hashlib.sha256(f"{url}|{position}".encode()).hexdigest()
    return digest[:16]


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into contiguous, gap-free sentence spans.

    Each span's trailing whitespace is absorbed into the sentence that
    precedes it, so concatenating ``text[start:end]`` for every span in
    order reproduces ``text`` exactly. That is what lets any contiguous
    run of spans be sliced straight back out of the source string.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.end()
        spans.append((start, end))
        start = end
    spans.append((start, len(text)))
    return spans


def _group_blocks(doc: ExtractedDoc) -> list[tuple[list[str], int, int]]:
    """Consecutive block index ranges (inclusive) sharing one heading_path."""
    groups: list[tuple[list[str], int, int]] = []
    if not doc.blocks:
        return groups
    group_start = 0
    for idx in range(1, len(doc.blocks)):
        if doc.blocks[idx].heading_path != doc.blocks[group_start].heading_path:
            groups.append((doc.blocks[group_start].heading_path, group_start, idx - 1))
            group_start = idx
    groups.append((doc.blocks[group_start].heading_path, group_start, len(doc.blocks) - 1))
    return groups


def _block_offsets(doc: ExtractedDoc) -> list[tuple[int, int]]:
    """Char offset of each block's text within ``doc.text`` (blocks joined
    with ``"\\n\\n"``, matching :attr:`ExtractedDoc.text` exactly)."""
    offsets: list[tuple[int, int]] = []
    pos = 0
    for block in doc.blocks:
        start = pos
        end = start + len(block.text)
        offsets.append((start, end))
        pos = end + 2  # length of the "\n\n" separator before the next block
    return offsets


def _overlap_start_idx(
    sentence_spans: list[tuple[int, int]],
    group_text: str,
    counter: TokenCounter,
    prev_first: int,
    prev_last: int,
    overlap_tokens: int,
) -> int:
    """First sentence index to carry into the next chunk as overlap.

    Walks backward from the previous chunk's last sentence, accumulating
    tokens until at least ``overlap_tokens`` are covered. Always stops
    strictly after ``prev_first``, so the whole previous chunk is never
    repeated (when the previous chunk is a single sentence, that means no
    overlap at all -- there is nothing shorter than the full chunk to take).
    """
    if overlap_tokens <= 0 or prev_last <= prev_first:
        return prev_last + 1
    total = 0
    idx = prev_last + 1
    for i in range(prev_last, prev_first, -1):
        start, end = sentence_spans[i]
        total += counter.count(group_text[start:end])
        idx = i
        if total >= overlap_tokens:
            break
    return idx


def _pack_group(
    sentence_spans: list[tuple[int, int]],
    group_text: str,
    counter: TokenCounter,
    settings: ChunkSettings,
) -> list[tuple[int, int]]:
    """Greedy, overlap-aware packing of sentence indices into chunk ranges.

    Returns a list of ``(first_sentence_idx, last_sentence_idx)`` pairs
    (inclusive), already merged for trailing fragments under
    ``settings.min_tokens``.
    """
    n = len(sentence_spans)
    ranges: list[tuple[int, int]] = []
    i = 0
    prev_range: tuple[int, int] | None = None
    while i < n:
        cur_first = (
            i
            if prev_range is None
            else _overlap_start_idx(
                sentence_spans,
                group_text,
                counter,
                prev_range[0],
                prev_range[1],
                settings.overlap_tokens,
            )
        )
        cur_last = cur_first
        j = cur_first
        while j < n:
            if j <= i:
                # Mandatory: either carried-over overlap content (j < i) or
                # the next never-yet-included sentence (j == i), which must
                # be admitted so the pack always makes forward progress.
                cur_last = j
                j += 1
                continue
            start = sentence_spans[cur_first][0]
            end = sentence_spans[j][1]
            if counter.count(group_text[start:end]) <= settings.target_tokens:
                cur_last = j
                j += 1
            else:
                break
        ranges.append((cur_first, cur_last))
        prev_range = (cur_first, cur_last)
        i = cur_last + 1

    if len(ranges) > 1:
        last_first, last_last = ranges[-1]
        start = sentence_spans[last_first][0]
        end = sentence_spans[last_last][1]
        if counter.count(group_text[start:end]) < settings.min_tokens:
            ranges[-2] = (ranges[-2][0], last_last)
            ranges.pop()
    return ranges


def chunk_document(
    doc: ExtractedDoc,
    settings: ChunkSettings,
    counter: TokenCounter | None = None,
) -> list[Chunk]:
    """Split a document into overlapping token-aware chunks.

    Guarantees:
      * no chunk exceeds ``target_tokens`` by more than one sentence;
      * consecutive chunks overlap by ~``overlap_tokens``;
      * trailing fragments under ``min_tokens`` merge backward;
      * ``start_char``/``end_char`` index into ``doc.text``;
      * heading_path of the chunk's first block is preserved.
    """
    if not doc.blocks:
        return []
    full_text = doc.text
    if not full_text.strip():
        return []

    active_counter: TokenCounter = counter if counter is not None else HeuristicTokenCounter()
    offsets = _block_offsets(doc)

    raw_chunks: list[tuple[list[str], int, int, str]] = []
    for heading_path, block_start, block_end in _group_blocks(doc):
        group_start_char = offsets[block_start][0]
        group_end_char = offsets[block_end][1]
        group_text = full_text[group_start_char:group_end_char]
        if not group_text.strip():
            continue
        spans = _sentence_spans(group_text)
        for first_idx, last_idx in _pack_group(spans, group_text, active_counter, settings):
            start_char = group_start_char + spans[first_idx][0]
            end_char = group_start_char + spans[last_idx][1]
            raw_chunks.append((heading_path, start_char, end_char, full_text[start_char:end_char]))

    chunks: list[Chunk] = []
    for position, (heading_path, start_char, end_char, text) in enumerate(raw_chunks):
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc.url, position),
                url=doc.url,
                doc_title=doc.title,
                heading_path=list(heading_path),
                text=text,
                token_count=active_counter.count(text),
                position=position,
                start_char=start_char,
                end_char=end_char,
            )
        )
    return chunks
