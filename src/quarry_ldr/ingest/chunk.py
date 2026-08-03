"""Token-aware chunking: ~512 tokens, 64 overlap, heading path preserved.

Token counting is pluggable via the TokenCounter protocol: runtime uses the
embedder's HF tokenizer, tests use the deterministic offline heuristic so the
suite needs no model downloads and no network.

Implemented in M4.
"""

from __future__ import annotations

import hashlib
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
    raise NotImplementedError
