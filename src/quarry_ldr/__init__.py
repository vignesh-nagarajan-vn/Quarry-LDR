"""Quarry-LDR: local deep research.

The local GPU compresses ~750K tokens of raw scraped text into ~60K tokens
of dense, deduplicated, reranked evidence, and by default it is also the
brain: an 8B model plans and writes the cited report, a 4B model triages
evidence and audits coverage, and a cross-encoder verifies every cited
sentence before render. ``engine.mode`` swaps the reasoning tier onto
Claude (assisted or premium) without touching the rest of the pipeline.
"""

__version__ = "1.0.0"
