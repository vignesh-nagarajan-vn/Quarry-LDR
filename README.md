# Quarry-LDR

Local deep research: a local GPU compresses the web into dense evidence, Claude reasons over it.

Repository: https://github.com/vignesh-nagarajan-vn/Quarry-LDR

Quarry-LDR takes a research topic and produces a cited markdown report through an iterative loop: plan, search, fetch, index, rerank, extract, find gaps, search again, synthesize. The design rests on one insight: the local GPU is a compression layer, not a brain. Its job is to turn roughly 750K tokens of raw scraped text into roughly 60K tokens of deduplicated, reranked evidence. The Anthropic API is then called on that small, high signal payload for the work that actually needs intelligence. Retrieval quality beats context stuffing, so the hybrid approach is both cheaper and better.

## Why this exists: the cost math

| Approach | What happens | Cost per report |
| --- | --- | --- |
| Naive (everything to the API) | ~750K raw tokens through Opus for triage and synthesis | $10 to $15 |
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

A realistic substantive report (3 loop iterations, ~300 sources) costs $1.00 to $1.50: one Opus plan call, a Sonnet gap call per iteration, and section-by-section Opus synthesis over a prompt-cached corpus. Caching matters: ten section calls over a 60K token corpus cost $0.87 with the 1 hour cache versus $3.00 without.

## Architecture

```
topic
  |
  v
[PLAN]        Opus 5, one call. 8 to 15 sub-questions, each with
              seed queries and a success criterion.
  |
  v
[SEARCH]      Local SearXNG in Docker. Free and unlimited.
  |
  v
[FETCH]       httpx, robots.txt respected, per-domain rate limit,
              content-addressed disk cache. Never fetch a URL twice.
  |
  v
[EXTRACT]     trafilatura. Boilerplate-free text plus heading structure.
  |
  v
[CHUNK]       Token-aware, ~512 tokens, 64 overlap, heading path kept.
  |
  v
[EMBED]       Local GPU. Batched. Thousands of chunks.
  |
  v
[DEDUP]       Local GPU. Cosine similarity plus SimHash shingling.
              Expect to drop 30 to 50 percent.
  |
  v
[RETRIEVE]    LanceDB ANN. Top 200 per sub-question.
  |
  v
[RERANK]      Local GPU cross-encoder. 200 candidates down to 40.
              The single highest leverage quality lever in the system.
  |
  v
[TRIAGE]      Local 4B model. Per-chunk structured JSON:
              {relevant, claim, evidence_span, confidence}
  |
  v
[GAP]         Sonnet 5. Coverage versus the plan's success criteria.
  |          |
  |          +---> loop back to SEARCH (max iterations, default 3)
  v
[SYNTHESIZE]  Opus 5, section by section, over a prompt-cached corpus.
  |
  v
[RENDER]      Markdown report with numbered citations resolving to
              URL plus chunk anchor, a cost ledger, and a run manifest.
```

All GPU residency is owned by a VRAM arbiter with a hard budget (default 6.5 GB): it loads models on demand, evicts by LRU, and serializes access so the embedder, reranker, and local LLM never fight over 8 GB. Stages are batched by model, never interleaved, because eviction and reload are expensive.

## Configuration reference

Defaults live in `config/default.yaml`; identical defaults are baked into the code so a missing file never breaks a run. Override with a `--config your.yaml` layer or `QUARRY_<SECTION>__<KEY>` environment variables.

| Key | Default | What changes if you move it |
| --- | --- | --- |
| `run.max_iterations` | 3 | More loops find more evidence, cost one Sonnet gap call each plus search time |
| `run.cost_cap_usd` | 5.0 | Hard API spend cap; the ledger raises mid-run when crossed |
| `run.data_dir` | `data` | Where cache, index, run DB, logs, and reports live |
| `run.models_dir` | `models` | Local weights and the llama.cpp server binary (`download_models.py`) |
| `models.plan` | `claude-opus-5` | Planning model; cheaper models produce weaker decompositions |
| `models.gap` | `claude-sonnet-5` | Runs every iteration, keep it cheap |
| `models.synthesize` | `claude-opus-5` | The one place maximum capability pays for itself |
| `models.extract_fallback` | `claude-haiku-4-5-20251001` | Bulk extraction fallback via Batch API |
| `models.embedder` | `BAAI/bge-m3` | Swap for `Qwen/Qwen3-Embedding-0.6B` to trade quality for VRAM |
| `models.reranker` | `BAAI/bge-reranker-v2-m3` | The quality lever; changing this changes evidence quality most |
| `models.triage_gguf_repo` / `_file` | Qwen3 4B Q4_K_M | Local triage model; must fit VRAM entirely |
| `gpu.vram_budget_mb` | 6656 | Hard arbiter budget; set to ~80 percent of your card's VRAM |
| `gpu.footprints_mb.*` | measured | Declared per-model VRAM; corrected by `scripts/bench_vram.py` |
| `gpu.embed_batch_size` | 32 | Bigger is faster until it OOMs |
| `gpu.rerank_batch_size` | 16 | Same trade as embed batch |
| `search.searxng_url` | `http://localhost:8888` | Point at any SearXNG with JSON enabled |
| `search.results_per_query` | 10 | More results per query, more fetching per iteration |
| `search.timeout_s` | 15.0 | Per search request |
| `fetch.user_agent` | QuarryLDR/0.1 (+repo URL) | Identifying UA; keep it honest |
| `fetch.per_domain_rps` | 1.0 | Politeness rate per domain |
| `fetch.max_concurrency` | 8 | Global concurrent fetches |
| `fetch.timeout_s` | 20.0 | Per fetch |
| `fetch.max_bytes` | 5000000 | Oversized responses are rejected |
| `fetch.cache_dir` | `data/cache/fetch` | Content-addressed body cache; safe to delete |
| `chunk.target_tokens` | 512 | Bigger chunks, fewer boundaries, coarser citations |
| `chunk.overlap_tokens` | 64 | Overlap between consecutive chunks |
| `chunk.min_tokens` | 48 | Trailing fragments below this merge backward |
| `dedup.cosine_threshold` | 0.92 | Lower drops more near-duplicates |
| `dedup.simhash_hamming_max` | 3 | Higher drops more near-duplicates |
| `retrieve.ann_top_k` | 200 | ANN candidates per sub-question |
| `retrieve.rerank_top_k` | 40 | Evidence kept per sub-question after rerank |
| `triage.context_tokens` | 8192 | llama-server context; bigger costs VRAM (KV cache) |
| `triage.port` | 8555 | llama-server port |
| `triage.max_retries` | 2 | Retries for malformed JSON from the local model |
| `triage.confidence_floor` | 0.3 | Verdicts below this are dropped |
| `api.max_retries` | 5 | Backoff retries on 429/529 |
| `api.retry_base_s` | 1.0 | Exponential backoff base with jitter |
| `api.cache_ttl` | `1h` | Prompt cache TTL for the synthesis corpus |
| `api.batch_poll_s` | 30.0 | Batch API poll interval |
| `report.min_sections` / `max_sections` | 4 / 12 | Report shape bounds |

## Hardware notes

Developed and tuned on a laptop RTX 4060 (8 GB, compute capability 8.9). Any CUDA GPU with capability 8.0 or newer works; adjust `gpu.vram_budget_mb` to about 80 percent of your VRAM. For Blackwell cards (RTX 50 series, `sm_120`), you need CUDA 12.8 or newer wheels; the project pins the cu128 PyTorch index, and `scripts/verify_gpu.py` proves a real matmul executes on device rather than trusting `torch.cuda.is_available()`.

Two constraints shape the design:

- Partial offload is catastrophic, not gradual. A model that does not fit entirely in VRAM decodes an order of magnitude slower over PCIe. The arbiter enforces full residency or refuses to load.
- Laptops throttle. Sustained multi-hour load runs well below burst benchmarks; actual tokens per second are logged so you can see it.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `verify_gpu.py` fails on capability | Wheels lack kernels for your GPU. For `sm_120` (Blackwell) install the cu128 index wheels: `uv sync --extra gpu` uses it already; check your driver supports CUDA 12.8. |
| SearXNG returns HTML instead of JSON | The `json` format is not enabled. Confirm `docker/searxng/settings.yml` lists `json` under `search.formats`, then `quarry searxng down && quarry searxng up`. |
| llama-server fails to start, port in use | Another process holds port 8555. Change `triage.port` or free the port. |
| `SchemaMismatchError` from LanceDB | The index was written by an older schema. Delete the index directory under `data/`; it is a rebuildable cache. |
| CUDA OOM under the arbiter | Declared footprints are lower than reality on your card. Run `scripts/bench_vram.py` and update `gpu.footprints_mb`, or lower batch sizes. |
| API 429 / 529 | The client retries with backoff up to `api.max_retries`. Persistent 429 means your account rate limits; lower concurrency or wait. |
| `make searxng` says Docker is missing | Install Docker Desktop (Windows/macOS) or Docker Engine (Linux) and start it. Everything except live search works without it. |

## Contributing

Read `CLAUDE.md` for the invariants (they are enforced by tests) and `COMMIT.md` for the commit contract. Run `make verify` before any PR: format, lint, type check, and the CPU-only test suite must pass without network or an API key.

## License

MIT. See `LICENSE`.
