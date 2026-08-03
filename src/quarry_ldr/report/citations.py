"""Numbered citations resolving to URL plus chunk anchor.

Every claim in a report carries a citation like [3]; every citation resolves
to a real URL and character offsets into the extracted document. A citation
that does not resolve is a CitationError, and a report containing one is a
test failure, not a warning.
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
        existing = self._number_by_chunk.get(chunk.chunk_id)
        if existing is not None:
            return existing
        number = len(self._by_number) + 1
        self._by_number[number] = Citation(
            number=number,
            chunk_id=chunk.chunk_id,
            url=chunk.url,
            title=chunk.doc_title,
            heading_path=list(chunk.heading_path),
            start_char=chunk.start_char,
            end_char=chunk.end_char,
        )
        self._number_by_chunk[chunk.chunk_id] = number
        return number

    def get(self, number: int) -> Citation:
        """Raise CitationError for unknown numbers."""
        citation = self._by_number.get(number)
        if citation is None:
            known = len(self._by_number)
            raise CitationError(f"citation [{number}] does not resolve ({known} citations exist)")
        return citation

    def __len__(self) -> int:
        return len(self._by_number)

    def all(self) -> list[Citation]:
        """All citations, ascending by number."""
        return [self._by_number[number] for number in sorted(self._by_number)]

    def numbers_in(self, markdown: str) -> list[int]:
        """All [n] markers appearing in a markdown string, in order."""
        return [int(match.group(1)) for match in CITATION_MARKER.finditer(markdown)]

    def validate_markdown(self, markdown: str) -> None:
        """Raise CitationError if any [n] in the text does not resolve."""
        for number in self.numbers_in(markdown):
            self.get(number)

    def references_markdown(self) -> str:
        """The numbered references section for the end of the report."""
        lines = ["## References", ""]
        for citation in self.all():
            heading = " > ".join(citation.heading_path) if citation.heading_path else ""
            anchor = f"chunk {citation.chunk_id}, chars {citation.start_char}-{citation.end_char}"
            title = citation.title or citation.url
            suffix = f" ({heading}; {anchor})" if heading else f" ({anchor})"
            lines.append(f"[{citation.number}] {title} - {citation.url}{suffix}")
        return "\n".join(lines)
