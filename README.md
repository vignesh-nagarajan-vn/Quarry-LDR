# Quarry-LDR

Local deep research: a local GPU compresses the web into dense evidence, Claude reasons over it.

Repository: https://github.com/vignesh-nagarajan-vn/Quarry-LDR

Quarry-LDR takes a research topic and produces a cited markdown report through an iterative loop: plan, search, fetch, index, rerank, extract, find gaps, search again, synthesize. The design rests on one insight: the local GPU is a compression layer, not a brain. Its job is to turn roughly 750K tokens of raw scraped text into roughly 60K tokens of deduplicated, reranked evidence. The Anthropic API is then called on that small, high signal payload for the work that actually needs intelligence. Retrieval quality beats context stuffing, so the hybrid approach is both cheaper and better.

## Why this exists: the cost math

| Approach | What happens | Cost per report |
| --- | --- | --- |
| Naive (everything to API) | ~750K raw tokens through Opus for triage and synthesis | $10 to $15 |
| Hybrid (Quarry-LDR) | Local GPU embeds, dedups, reranks, triages; API sees ~60K curated tokens | under $1 |

Equal or better output at roughly a tenth of the cost, because the API only ever sees evidence worth reasoning about.

## Quickstart

Prerequisites: an NVIDIA GPU (8 GB VRAM or more recommended), Docker Desktop or Docker Engine for SearXNG, and an Anthropic API key. `uv`, GNU make, and Python 3.12 are installed by bootstrap if missing.

```bash
git clone https://github.com/vignesh-nagarajan-vn/Quarry-LDR
cd Quarry-LDR
make bootstrap
cp .env.example .env          # paste your ANTHROPIC_API_KEY
make searxng                  # starts local search in Docker
uv run python scripts/download_models.py   # fetches llama-server and the triage GGUF
uv run quarry verify          # preflight check with remediation hints
uv run quarry research "your topic"
```

The report lands in `data/reports/`, with a cost ledger and a run manifest appended. `quarry resume <run_id>` continues an interrupted run; `quarry inspect <run_id>` dumps stage-by-stage state.

## API pricing and per-report cost

Prices in USD per million tokens, used verbatim by the cost ledger:

| Model | Input | Output | Batch in | Batch out | 1h cache write | Cache read |
| --- | --- | --- | --- | --- | --- | --- |
| `claude-opus-5` | $5 | $25 | $2.50 | $12.50 | $10 | $0.50 |
| `claude-sonnet-5` | $2 | $10 | $1.00 | $5.00 | $4 | $0.20 |
| `claude-haiku-4-5-20251001` | $1 | $5 | $0.50 | $2.50 | $2 | $0.10 |

Two caveats the ledger encodes:

- Sonnet 5 pricing above is introductory through August 31, 2026, then becomes $3 in, $15 out. The pricing table is date aware, so ledgers stay accurate after the change.
- Claude 4.7 and later use a tokenizer that produces roughly 30 percent more tokens for the same text. Costs are always computed from the `usage` block the API returns, never from character counts.

A realistic substantive report (3 loop iterations, ~300 sources) costs $1.00 to $1.50. That buys one Opus plan call, a Sonnet gap call per iteration, and section-by-section Opus synthesis over a prompt-cached corpus. Caching matters: ten section calls over a 60K token corpus cost $0.87 with the 1 hour cache versus $3.00 without.

## How it works

A 14-stage checkpointed pipeline: one Opus call plans sub-questions, SearXNG and a polite fetcher gather sources for free, the local GPU embeds, deduplicates, reranks, and triages them down to dense evidence, a Sonnet gap check decides whether to loop, and Opus writes the report section by section over one prompt-cached corpus. A VRAM arbiter with a hard 6.5 GB budget owns all GPU residency so three models share one 8 GB card safely. The full diagram, the arbiter rules, and every configuration key are in [docs/Architecture.md](docs/Architecture.md).

## Hardware notes

Developed and tuned on a laptop RTX 4060 (8 GB, compute capability 8.9). Any CUDA GPU with capability 8.0 or newer works; set `gpu.vram_budget_mb` to about 80 percent of your card's VRAM. For Blackwell cards (RTX 50 series, `sm_120`) you need CUDA 12.8 or newer wheels; the project pins the cu128 PyTorch index. Constraints and details are in [docs/Architecture.md](docs/Architecture.md).

## Repo map

```
Quarry-LDR/
  src/quarry_ldr/
    gpu/                VRAM arbiter, embedder, reranker, llama-server client
    ingest/             search, fetch, extract, chunk, dedup
    index/              LanceDB vector store with a versioned schema
    pipeline/           plan, retrieve, triage, gap, synthesize, orchestrator
    providers/          the Anthropic client: caching, retries, batch, ledger hooks
    report/             citations and markdown rendering
    cli.py              the quarry command
    state.py            SQLite run store; every stage transition is a row
    ledger.py           date-aware pricing, costs from API usage blocks only
  config/default.yaml   documented defaults, mirrored in code
  docker/               SearXNG compose with the JSON API enabled
  docs/                 architecture, configuration, troubleshooting
  scripts/              bootstrap, model download, GPU verify, VRAM bench,
                        fixture generator, pre-public audit, smoke test
  tests/                CPU-only suite over a synthetic 40-document corpus
```

## Documentation

- [docs/Architecture.md](docs/Architecture.md): pipeline diagram, VRAM arbiter, hardware design target, full configuration reference.
- [docs/Troubleshooting.md](docs/Troubleshooting.md): symptoms, causes, and exact fixes.
- `CLAUDE.md`: invariants and operating rules for coding sessions.
- `COMMIT.md`: the commit contract.
- `DECISIONS.md`: every design decision and measured deviation.

## Contributing

Read `CLAUDE.md` for the invariants (they are enforced by tests) and `COMMIT.md` for the commit contract. Run `make verify` before any PR: format, lint, type check, and the CPU-only test suite must pass without network or an API key.

## License

MIT. See `LICENSE`.
