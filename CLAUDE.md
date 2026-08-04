# CLAUDE.md

Operating instructions for Claude Code sessions in this repo. Terse on purpose.

## Purpose

Quarry-LDR turns a research topic into a cited markdown report. A local GPU compresses ~750K tokens of scraped web text into ~60K tokens of deduplicated, reranked evidence; the Anthropic API does planning, gap analysis, and synthesis on that small payload. Measured cost: $1.36 to $2.88 per report on the claude 5 models (README pricing section has the breakdown).

## Architecture map

| Path | Owns |
| --- | --- |
| `src/quarry_ldr/config.py` | Layered settings: defaults < default.yaml < user yaml < env |
| `src/quarry_ldr/logging.py` | structlog JSON file + console, `redact()` on every event |
| `src/quarry_ldr/state.py` | SQLite run store; every stage transition is a row; resume |
| `src/quarry_ldr/ledger.py` | Date-aware pricing, cost from API `usage` blocks only |
| `src/quarry_ldr/gpu/arbiter.py` | All GPU residency: hard budget, LRU eviction, async lock |
| `src/quarry_ldr/gpu/` | embedder, reranker, llama-server lifecycle + local client |
| `src/quarry_ldr/ingest/` | search (SearXNG), fetch (robots, rate limit, cache), extract, chunk, dedup |
| `src/quarry_ldr/index/` | LanceDB store with versioned schema |
| `src/quarry_ldr/pipeline/` | Stage functions + orchestrator state machine (`run.py`) |
| `src/quarry_ldr/providers/anthropic_client.py` | The only API surface: retries, caching, batch, ledger hooks |
| `src/quarry_ldr/report/` | Render + citations; every claim resolves to URL + chunk anchor |
| `scripts/` | bootstrap, model download, GPU verify, VRAM bench, fixtures, audit, smoke |

## Invariants, never break these

1. All GPU model access goes through the arbiter. No exceptions, no direct `.to("cuda")`.
2. The ledger is updated from the API `usage` block, never estimated from text length.
3. Prompt cache prefixes are byte stable, asserted by hash; drift raises, it does not pay.
4. Every claim in a report carries a citation that resolves to a URL and chunk offsets.
5. Pipeline stages are batched by model: embed everything, then rerank, then triage. Never interleave GPU models.
6. No secrets, personal paths, or real scraped content in the repo or its history. Fixtures are synthetic.
7. The default test selection runs on CPU with no network and no API key, always.
8. Measured-vs-declared VRAM corrections reject a measurement under 25 percent of the declared footprint as implausible and keep the declared value (Windows WDDM hides child-process VRAM from `mem_get_info`).

## Commands

| Command | Does |
| --- | --- |
| `make bootstrap` | Install uv if needed, sync deps (GPU extra when NVIDIA present), install hooks |
| `make test` | CPU-only suite, network blocked, coverage gate 80 percent |
| `make verify` | format check, lint, mypy, tests: the milestone gate |
| `make fmt` | ruff format + autofix |
| `make lint` | ruff check + mypy |
| `make searxng` | Start SearXNG in Docker (remediation message if Docker absent) |
| `make searxng-down` | Stop the SearXNG container |
| `make smoke` | Real end-to-end run, $2.00 cap, prints ledger |
| `make audit` | Pre-public go/no-go: history secrets, paths, fixtures, em dashes |
| `make fixtures` | Regenerate the synthetic corpus deterministically |

## Conventions

- ruff for format and lint, mypy for types, full type hints everywhere.
- pydantic models at every boundary; parse, do not pass dicts around.
- Async by default in pipeline code; sync only where a library forces it (LanceDB), wrapped in `asyncio.to_thread`.
- Structured logging with `run_id` and `stage` bound on every event; `redact()` already runs on every sink, keep it that way.
- Windows, Linux, and WSL2 all work: pathlib everywhere, no shell-isms in library code.

## Cost discipline

Any change touching an API call path must state the token and dollar impact in the commit body, measured, not guessed. The ledger and its tests are the source of truth for pricing math.

## Session guidance

- Fan-out suits module-parallel work with frozen interfaces (the ingest and index layers, report/scripts/docs polish). Orchestration, the arbiter, the provider, and the state machine are sequential work; do them in one context.
- Keep subagents on cheaper models (Sonnet or Haiku). The orchestrating model reviews and judges; it does not type boilerplate.
- Read files, run tests, and grep via delegated agents when context is tight.

## Adding a new pipeline stage

1. Add the stage to `Stage` in `state.py` in execution order.
2. Write the stage function in `pipeline/` taking injected components, returning pydantic models.
3. Wire it into the orchestrator with `start_stage` / `complete_stage` checkpoints.
4. If it calls the API: route through `AnthropicProvider`, pass `stage=` for the ledger.
5. If it touches the GPU: register a `ModelSpec`, access only via `arbiter.acquire`.
6. Add unit tests against fixtures; update `quarry inspect` expectations.
7. Update the docs/Architecture.md diagram and this map if a new module appeared.

## Do not

- No LangChain, LangGraph, or orchestration frameworks.
- No bare `except:`; catch what you mean.
- No `print` outside `cli.py` and `scripts/` (ruff T20 enforces this).
- No network in unit tests (pytest-socket enforces this).
- No new dependency without a line in `DECISIONS.md`.
- No model IDs hardcoded in business logic; they live in config.

## Before this repo goes public

1. `make verify` green on a clean clone.
2. CI green on the default selection.
3. README quickstart re-tested from scratch.
4. No badges pointing at private resources.
5. `make audit` returns go. This is the final gate.
