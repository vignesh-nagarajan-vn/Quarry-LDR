"""Boilerplate-free text extraction with heading structure, via trafilatura.

The XML output mode preserves document structure; headings become the
``heading_path`` carried by every block (and later every chunk and citation).

Implemented in M4.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import trafilatura
from pydantic import BaseModel, Field

from quarry_ldr.logging import get_logger

logger = get_logger()

# Below this, a document is treated as having nothing useful to index
# (paywall teasers, JS-only shells that trafilatura barely scrapes).
MIN_DOC_TEXT_CHARS = 200

# Blocks shorter than this are folded into the previous block rather than
# standing alone (stray captions, single-sentence fragments).
MIN_BLOCK_CHARS = 40

_TITLE_TAG_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")

# Tags that introduce a new nesting level in trafilatura's XML output whose
# rend attribute we don't recognise (or that carries none) default to this
# level, i.e. treated like a normal section heading rather than the title.
_DEFAULT_HEADING_LEVEL = 2


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


def _decode(html: bytes | str) -> str:
    """Bytes decode utf-8 strict, falling back to lossy replacement so a
    malformed document extracts instead of raising."""
    if isinstance(html, str):
        return html
    try:
        return html.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("extract.decode_replaced", reason="invalid_utf8")
        return html.decode("utf-8", errors="replace")


def _normalize_ws(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _title_from_html(html: str) -> str:
    """Last-resort title: the literal ``<title>`` tag text."""
    match = _TITLE_TAG_RE.search(html)
    if not match:
        return ""
    return _normalize_ws(match.group(1))


def _heading_level(rend: str | None) -> int:
    if rend and len(rend) == 2 and rend[0] == "h" and rend[1].isdigit():
        return int(rend[1])
    return _DEFAULT_HEADING_LEVEL


def _element_text(elem: ET.Element) -> str:
    return _normalize_ws("".join(elem.itertext()))


def _row_text(elem: ET.Element) -> str:
    """A table row's cells joined with a separator; falls back to the row's
    full text if it carries no ``cell`` children."""
    cells = [child for child in elem if child.tag == "cell"]
    if not cells:
        return _element_text(elem)
    parts = [_element_text(cell) for cell in cells]
    return " | ".join(part for part in parts if part)


def _walk(
    elem: ET.Element,
    heading_stack: list[tuple[int, str]],
    raw_blocks: list[tuple[list[str], str]],
) -> None:
    """Depth-first walk collecting (heading_path, text) pairs.

    ``head`` elements update the heading stack (a heading pops any stack
    entries at its level or deeper, then pushes itself, so a same-level head
    replaces the current tail). The level-1 head is trafilatura's document
    title and is deliberately excluded from the path. ``p``/``item``/``row``
    elements become blocks and are not descended into further (their full
    nested text is already captured). Everything else (``main``, ``list``,
    ``table``, ...) is a pure container we recurse through.
    """
    for child in elem:
        tag = child.tag
        if tag == "head":
            level = _heading_level(child.get("rend"))
            text = _element_text(child)
            if text and level > 1:
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, text))
        elif tag in ("p", "item"):
            text = _element_text(child)
            if text:
                raw_blocks.append(([h for _, h in heading_stack], text))
        elif tag == "row":
            text = _row_text(child)
            if text:
                raw_blocks.append(([h for _, h in heading_stack], text))
        else:
            _walk(child, heading_stack, raw_blocks)


def _merge_tiny_blocks(raw_blocks: list[tuple[list[str], str]]) -> list[Block]:
    merged: list[Block] = []
    for heading_path, text in raw_blocks:
        if merged and len(text) < MIN_BLOCK_CHARS:
            previous = merged[-1]
            merged[-1] = Block(
                heading_path=previous.heading_path,
                text=f"{previous.text} {text}".strip(),
            )
        else:
            merged.append(Block(heading_path=list(heading_path), text=text))
    return merged


def extract_document(html: bytes | str, url: str) -> ExtractedDoc | None:
    """Extract main content. Returns None when nothing useful survives
    (JS-only shells, paywall stubs, boilerplate-only pages)."""
    text_html = _decode(html)

    xml_output = trafilatura.extract(
        text_html,
        url=url,
        output_format="xml",
        include_comments=False,
        include_tables=True,
    )
    if not xml_output:
        return None

    try:
        root = ET.fromstring(xml_output)
    except ET.ParseError:
        logger.warning("extract.xml_parse_failed", url=url)
        return None

    main_elem = root.find("main")
    if main_elem is None:
        main_elem = root

    raw_blocks: list[tuple[list[str], str]] = []
    _walk(main_elem, [], raw_blocks)
    blocks = _merge_tiny_blocks(raw_blocks)

    total_chars = sum(len(block.text) for block in blocks)
    if total_chars < MIN_DOC_TEXT_CHARS:
        return None

    title = ""
    lang: str | None = None
    published: str | None = None
    metadata = trafilatura.extract_metadata(text_html, default_url=url)
    if metadata is not None:
        title = metadata.title or ""
        lang = metadata.language or None
        published = metadata.date or None
    if not title:
        title = _title_from_html(text_html)

    return ExtractedDoc(
        url=url,
        title=title,
        blocks=blocks,
        lang=lang,
        published=published,
    )
