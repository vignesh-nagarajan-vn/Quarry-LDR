"""External model providers.

Two, both satisfying the ``Provider`` protocol in ``base``: the Anthropic
API client and the local llama-server client. The orchestrator routes
PLAN/GAP/SYNTHESIZE per ``engine.mode``; the ledger and cache-prefix rules
apply to both. The original design was API-only ("no alternative providers
by design"); v1 amended it, see DECISIONS.md.
"""
