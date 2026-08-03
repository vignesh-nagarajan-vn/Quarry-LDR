"""Boilerplate-free text extraction with heading structure, via trafilatura.

The XML output mode preserves document structure; headings become the
``heading_path`` carried by every block (and later every chunk and citation).

Implemented in M4.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Block(BaseModel):
    """A run of body text under a heading path like ["Results", "Latency"]."""

    heading_path: list[str] = Field(default_factory=list)
    text: str


class ExtractedDoc(BaseModel):
    url: str
    title: str = ""
    blocks: list[Block] = Field(default_factory=list)
    lang: str | None = None
    published: str | None = None

    @property
    def text(self) -> str:
        """All block text joined; the character-offset space chunks index into."""
        return "\n\n".join(block.text for block in self.blocks)


def extract_document(html: bytes | str, url: str) -> ExtractedDoc | None:
    """Extract main content. Returns None when nothing useful survives
    (JS-only shells, paywall stubs, boilerplate-only pages)."""
    raise NotImplementedError
