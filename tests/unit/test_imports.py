"""Every module in the package imports cleanly and exposes its public surface.

This is the M0 contract test: parallel milestone agents build against these
names, so their existence is enforced from the first commit.
"""

from __future__ import annotations

import importlib
import pkgutil

import quarry_ldr

EXPECTED_PUBLIC = {
    "quarry_ldr.config": ["QuarryConfig", "load_config", "deep_merge"],
    "quarry_ldr.logging": ["setup_logging", "redact", "get_logger"],
    "quarry_ldr.state": ["RunStore", "RunRecord", "StageRecord", "Stage", "StageStatus"],
    "quarry_ldr.ledger": ["Ledger", "TokenUsage", "PRICING", "price_for", "compute_cost"],
    "quarry_ldr.gpu.arbiter": ["VramArbiter", "ModelSpec", "BudgetExceededError", "GpuBackend"],
    "quarry_ldr.gpu.embedder": ["Embedder"],
    "quarry_ldr.gpu.reranker": ["Reranker", "ScoredChunk"],
    "quarry_ldr.gpu.local_llm": ["LlamaServer", "LocalLLM", "LlamaServerError"],
    "quarry_ldr.ingest.search": ["SearxClient", "SearchResult", "SearxngError"],
    "quarry_ldr.ingest.fetch": ["Fetcher", "FetchResult", "FetchStatus", "normalize_url"],
    "quarry_ldr.ingest.extract": ["extract_document", "ExtractedDoc", "Block"],
    "quarry_ldr.ingest.chunk": ["chunk_document", "Chunk", "TokenCounter", "HeuristicTokenCounter"],
    "quarry_ldr.ingest.dedup": ["dedup_chunks", "simhash64", "hamming", "DedupResult"],
    "quarry_ldr.index.schema": ["chunk_arrow_schema", "SCHEMA_VERSION", "SchemaMismatchError"],
    "quarry_ldr.index.store": ["VectorStore", "RetrievedChunk"],
    "quarry_ldr.pipeline.plan": ["make_plan", "ResearchPlan", "SubQuestion"],
    "quarry_ldr.pipeline.retrieve": ["retrieve_candidates", "rerank_candidates"],
    "quarry_ldr.pipeline.triage": ["triage_chunks", "TriageVerdict", "TriagedChunk"],
    "quarry_ldr.pipeline.gap": ["analyze_gaps", "GapAnalysis", "CoverageAssessment"],
    "quarry_ldr.pipeline.synthesize": ["synthesize", "build_evidence_corpus", "DraftReport"],
    "quarry_ldr.pipeline.run": ["Orchestrator", "RunResult"],
    "quarry_ldr.providers.anthropic_client": [
        "AnthropicProvider",
        "CompletionResult",
        "CachePrefixError",
        "hash_corpus",
    ],
    "quarry_ldr.report.citations": ["CitationIndex", "Citation", "CitationError"],
    "quarry_ldr.report.render": ["render_report", "write_report", "RunManifest"],
}


def test_all_modules_import() -> None:
    prefix = quarry_ldr.__name__ + "."
    found = []
    for module_info in pkgutil.walk_packages(quarry_ldr.__path__, prefix):
        importlib.import_module(module_info.name)
        found.append(module_info.name)
    assert "quarry_ldr.cli" in found
    assert "quarry_ldr.pipeline.run" in found


def test_public_surfaces() -> None:
    for module_name, names in EXPECTED_PUBLIC.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert hasattr(module, name), f"{module_name} is missing {name}"
