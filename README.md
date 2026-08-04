# Quarry-LDR

Local deep research: a local GPU compresses the web into dense evidence, Claude reasons over it.

Repository: https://github.com/vignesh-nagarajan-vn/Quarry-LDR

Quarry-LDR takes a research topic and produces a cited markdown report through an iterative loop: plan, search, fetch, index, rerank, extract, find gaps, search again, synthesize. The design rests on one insight: the local GPU is a compression layer, not a brain. Its job is to turn roughly 750K tokens of raw scraped text into roughly 60K tokens of deduplicated, reranked evidence. The Anthropic API is then called on that small, high signal payload for the work that actually needs intelligence. Retrieval quality beats context stuffing, so the hybrid approach is both cheaper and better.

## Deployment status (2026-08-04, remove this section before going public)

Blackwell validation is done on the target RTX 5060 Mobile laptop: `verify_gpu.py` passes, all gpu-marked tests pass, measured VRAM footprints are in config, and the verify gate is green. No end-to-end report has been produced yet. Remaining, in order:

Manual steps:

- [x] Install Docker Desktop with the WSL2 backend (SearXNG needs it): done, Docker Desktop 4.85.0 per user plus WSL 2.7.11; engine startup needed the Docker AI feature disabled (unix socket creation fails on this Windows build)
- [x] Paste the real `ANTHROPIC_API_KEY` into `.env` (a placeholder is there now)

Then prompt Claude Code to:

- [x] Start SearXNG (`quarry searxng up`) and confirm the JSON API answers on localhost:8888
- [x] Run `scripts/smoke.py` (hard $2.00 cap): done, run 08ae0cec1ee3 passed with resolvable citations at $1.36 measured (18 sources, 6 sections, 64K-token cached corpus)
- [x] Run one full research run on a topic you give it, then reconcile this README's pricing section against the measured ledger: done, run 1497b2907a55 (3 iterations, 10 sections, $2.88 measured; pricing section updated to measured values)
- [x] Interrupt a run, then exercise `quarry resume <run_id>` and `quarry inspect <run_id>`: done, the run above was killed mid-triage and resumed; 21 completed stages replayed from persisted payloads in 49 ms with no recomputation and no new ledger rows
- [x] Finish: clean-clone `make verify`, CI green, `make audit` GO, and a DECISIONS.md environment entry for this machine: all done 2026-08-04 (clean clone of f439e89 passed format, lint, mypy, 292 tests, and audit GO; CI green on every push; environment entry in DECISIONS.md)

## Why this exists: the cost math

| Approach | What happens | Cost per report |
| --- | --- | --- |
| Naive (All API) | ~750K raw tokens through Opus for triage and synthesis | $10 to $15 |
| Hybrid (Quarry-LDR) | Local GPU embeds, dedups, reranks, triages; ~60K tokens to API | $1.36 to $2.88 measured |

Equal or better output at roughly a tenth of the cost, because the API only ever sees evidence worth reasoning about.

## Quickstart

Prerequisites: an NVIDIA GPU (8 GB VRAM or more recommended), Docker Desktop or Docker Engine for SearXNG, and an Anthropic API key. `uv`, GNU make, and Python 3.12 are installed by bootstrap if missing.

```bash
git clone https://github.com/vignesh-nagarajan-vn/Quarry-LDR
cd Quarry-LDR
make bootstrap                # fresh Windows without GNU make: powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
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

Measured on the first live runs (2026-08-04): a substantive report (3 loop iterations, 10 sections, ~60K-token cached corpus) cost $2.88; the single-iteration 6-section smoke run cost $1.36. That buys one Opus plan call, a Sonnet gap call per iteration, and section-by-section Opus synthesis over a prompt-cached corpus. The original $1.00 to $1.50 projection predates claude-opus-5, whose default adaptive thinking roughly doubles output-token spend. Caching matters and measured almost exactly as projected: corpus transport for ten section calls cost $0.84 with the 1 hour cache versus $2.89 if each call re-sent the corpus uncached.

## How it works

A 14-stage checkpointed pipeline: one Opus call plans sub-questions, SearXNG and a polite fetcher gather sources for free, the local GPU embeds, deduplicates, reranks, and triages them down to dense evidence, a Sonnet gap check decides whether to loop, and Opus writes the report section by section over one prompt-cached corpus. A VRAM arbiter with a hard 6.5 GB budget owns all GPU residency so three models share one 8 GB card safely. The full diagram, the arbiter rules, and every configuration key are in [docs/Architecture.md](docs/Architecture.md).

## Hardware target

Designed for a laptop NVIDIA RTX 5060 Mobile: 8 GB GDDR7, Blackwell architecture, `sm_120`. Blackwell needs CUDA 12.8 or newer kernels, so the project pins the cu128 PyTorch wheel index and `scripts/verify_gpu.py` proves a real matmul executes on device. Any CUDA GPU with compute capability 8.0 or newer also works; set `gpu.vram_budget_mb` to about 80 percent of your card's VRAM. Constraints and details are in [docs/Architecture.md](docs/Architecture.md).

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
