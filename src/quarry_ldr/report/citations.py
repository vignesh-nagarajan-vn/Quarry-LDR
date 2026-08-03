"""Numbered citations resolving to URL plus chunk anchor.

Every claim in a report carries a citation like [3]; every citation resolves
to a real URL and character offsets into the extracted document. A citation
that does not resolve is a CitationError, and a report containing one is a
test failure, not a warning.

Implemented in M10 (used from M8 onward via this interface).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from quarry_ldr.ingest.chunk import Chunk

CITATION_MARKER = re.compile(r"\[(\d+)\]")


class Citation(BaseModel):
    number: int
    chunk_id: str
    url: str
    title: str = ""
    heading_path: list[str] = Field(default_factory=list)
    start_char: int
    end_char: int


class CitationError(Exception):
    """A [n] marker with no matching citation, or an unresolvable citation."""


class CitationIndex:
    """Stable chunk -> citation-number assignment for one run."""

    def __init__(self) -> None:
        self._by_number: dict[int, Citation] = {}
        self._number_by_chunk: dict[str, int] = {}

    def add(self, chunk: Chunk) -> int:
        """Assign (or return the existing) citation number for a chunk."""
        raise NotImplementedError

    def get(self, number: int) -> Citation:
        """Raise CitationError for unknown numbers."""
        raise NotImplementedError

    def all(self) -> list[Citation]:
        """All citations, ascending by number."""
        raise NotImplementedError

    def numbers_in(self, markdown: str) -> list[int]:
        """All [n] markers appearing in a markdown string, in order."""
        raise NotImplementedError

    def validate_markdown(self, markdown: str) -> None:
        """Raise CitationError if any [n] in the text does not resolve."""
        raise NotImplementedError

    def references_markdown(self) -> str:
        """The numbered references section for the end of the report."""
        raise NotImplementedError
