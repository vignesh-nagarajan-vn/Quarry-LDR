"""Ingest layer: search, fetch, extract, chunk, dedup.

Data flows SearchResult -> FetchResult -> ExtractedDoc -> Chunk -> DedupResult.
Each module is independent and testable against the synthetic fixture corpus.
"""
