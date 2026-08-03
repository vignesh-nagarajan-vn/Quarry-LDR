"""Quarry-LDR: hybrid local/API deep research.

The local GPU is a compression layer, not a brain: it turns ~750K tokens of
raw scraped text into ~60K tokens of dense, deduplicated, reranked evidence.
The Anthropic API is then called on that small, high-signal payload for the
work that actually needs intelligence.
"""

__version__ = "0.1.0"
