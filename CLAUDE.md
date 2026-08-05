# CLAUDE.md

The complete operating context for anyone, human or agent, working in this repo: what the system is, how it is laid out, the invariants that must never break, the commit contract, and the working conventions. Read this before touching code.

## Purpose

Quarry-LDR turns a research topic into a cited report (markdown plus a branded PDF). A local GPU compresses ~750K tokens of scraped web text into ~60K tokens of deduplicated, reranked evidence, and `engine.mode` decides who reasons over it: `local` (default, Qwen3 8B/4B via llama-server, $0.00, no API key), `assisted` (local draft, Haiku gap checks and polish, measured $0.02 on the smoke rehearsal), or `premium` (Claude plans, audits, writes; measured $1.36 to $2.88 per report). VERIFY scores every cited sentence against its cited chunks in every mode and rewrites or drops what the evidence does not support.

The design bet, proven by the measured numbers: compression is the expensive part of research and a consumer GPU does it free; reasoning over compressed evidence is cheap enough to run locally by default and to buy selectively when quality demands it.

## Orientation

Read in this order when new to the repo:

1. [README.md](README.md): what it is, the engine table, the branch/version table.
2. [docs/Architecture.md](docs/Architecture.md): the pipeline, engine routing, VRAM arbiter, every config key.
3. [docs/RunGuide.md](docs/RunGuide.md): how to actually drive it, per engine.
4. [DECISIONS.md](DECISIONS.md): why everything is the way it is, with measured numbers. Any change that contradicts a DECISIONS entry needs a new entry, not a silent override.
5. This file's invariants and contracts, which bind every change.

The v0 hybrid design is preserved on branch `archive/v0-hybrid-api` (released as v0.9.0-beta); `main` is the v1 local-first line.

## Architecture Map

| Path | Owns |
| --- | --- |
| `src/quarry_ldr/config.py` | Layered settings: defaults < default.yaml < user yaml < env |
| `src/quarry_ldr/logging.py` | structlog JSON file + console, `redact()` on every event |
| `src/quarry_ldr/state.py` | SQLite run store; every stage transition is a row; resume |
| `src/quarry_ldr/ledger.py` | Date-aware pricing, cost from API `usage` blocks only |
| `src/quarry_ldr/gpu/arbiter.py` | All GPU residency: hard budget, LRU eviction, async lock |
| `src/quarry_ldr/gpu/` | embedder, reranker, llama-server lifecycle (triage + synth specs) + local client |
| `src/quarry_ldr/ingest/` | search (SearXNG), fetch (robots, rate limit, cache), extract, chunk, dedup |
| `src/quarry_ldr/index/` | LanceDB store with versioned schema |
| `src/quarry_ldr/pipeline/` | Stage functions + orchestrator state machine (`run.py`) |
| `src/quarry_ldr/pipeline/verify.py` | VERIFY: cross-encoder entailment gate + 4B rewrite-or-drop |
| `src/quarry_ldr/providers/base.py` | `Provider` protocol both backends satisfy; cache-prefix rules |
| `src/quarry_ldr/providers/anthropic_client.py` | The API backend: retries, caching, batch, ledger hooks |
| `src/quarry_ldr/providers/local_client.py` | The llama-server backend: same protocol, $0 ledger rows |
| `src/quarry_ldr/report/` | Render + citations; every claim resolves to URL + chunk anchor |
| `src/quarry_ldr/report/pdf.py`, `charts.py` | Branded Typst PDF + matplotlib run charts, fail-soft |
| `src/quarry_ldr/cli.py` | Typer CLI: research, resume, inspect, runs, verify, searxng |
| `scripts/` | bootstrap, model download, GPU verify, VRAM bench, fixtures, audit, smoke |
| `config/default.yaml` | Every default, mirrored byte-for-meaning in config.py (a drift test enforces it) |
| `docker/` | SearXNG compose file and settings (JSON format enabled) |
| `tests/` | `unit/` (CPU, offline, the default selection), `integration/` (gpu/live marked), `fixtures/` (synthetic) |
| `pdf-reports/` | Tracked sample PDFs with their measured numbers; `data/` stays runtime-only and gitignored |
| `docs/first-test/` | The v0 live validation record: cost anatomy and the $2.88 example report |

## Invariants, never break these

1. All GPU model access goes through the arbiter. No exceptions, no direct `.to("cuda")`.
2. The ledger is updated from usage blocks only, never estimated from text length: the API's `usage` for Claude calls, llama-server's OpenAI-compatible `usage` at zero price for `local/` model ids.
3. Prompt cache prefixes are byte stable, asserted by hash; drift raises, it does not pay.
4. Every claim in a report carries a citation that resolves to a URL and chunk offsets, and VERIFY gates it against the cited text before render.
5. Pipeline stages are batched by model: embed everything, then rerank, then triage. Never interleave GPU models; the synth 8B is resident only alone.
6. No secrets, personal paths, or real scraped content in the repo or its history. Fixtures are synthetic (`.example` URLs, provenance marker). The tracked sample reports in `pdf-reports/` are pipeline *output*, which is fine; raw fetched corpora are not.
7. The default test selection runs on CPU with no network and no API key, always. GPU and live tests are opt-in via `-m gpu` / `-m live`.
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
| `make smoke` | Real end-to-end run on the configured engine, $2.00 cap, prints ledger |
| `make smoke-local` | The same rehearsal forced to the local engine: $0, no API key |
| `make audit` | Pre-public go/no-go: history secrets, paths, fixtures, em dashes |
| `make fixtures` | Regenerate the synthetic corpus deterministically |

GPU-marked tests run as `uv run python -m pytest -m gpu <path> --no-cov` (module form works on machines where app-control policies block venv exe shims; prefer it in automation).

## Commit Contract

History is written for a public repo: no personal detail, no internal paths, no secrets, ever.

**Format.** Conventional Commits with the enforced type set `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`, `build`, `ci`. Scope is a module path segment: `feat(gpu): ...`, `fix(ingest): ...`, `chore(repo): ...` for cross-cutting. Subject in imperative mood, lowercase, no trailing period, 72 characters maximum. Footer ties the commit to its milestone: `Refs: M<n>`.

**Body.** Required for anything touching GPU memory, API cost, or the citation path. State measured impact, not intent: "Reduces synthesis cost from $3.00 to $0.87 on the 10-section fixture run" beats "improves caching". Cost discipline is a hard rule: any change touching an API call path must state the token and dollar impact in the commit body, measured, not guessed. The ledger and its tests are the source of truth for pricing math.

**Granularity.** One logical change per commit. A milestone is one commit unless it exceeds roughly 400 changed lines, then split along module boundaries.

**Pre-commit checklist.** Tests green (`make test`); `make verify` clean; secret scan clean (pre-commit runs detect-secrets); docs updated if behavior changed; `DECISIONS.md` updated if a dependency or design choice changed.

A good example:

```
feat(gpu): enforce hard VRAM budget with LRU eviction in arbiter

Budget violations are now impossible: acquiring a third model whose
footprint would exceed 6656 MB evicts the least recently used resident
first. Measured on the fake backend: peak declared residency 6200 MB
across the embed->rerank->triage swap sequence, previously unbounded.

Refs: M2
```

A bad one: `Fixed some GPU stuff and updated docs (WIP)`. No type or scope, past tense, vague, bundles unrelated changes, uncommitted-quality marker, no milestone, no measured impact.

## Code Conventions

- ruff for format and lint, mypy for types, full type hints everywhere.
- pydantic models at every boundary; parse, do not pass dicts around.
- Async by default in pipeline code; sync only where a library forces it (LanceDB), wrapped in `asyncio.to_thread`.
- Structured logging with `run_id` and `stage` bound on every event; `redact()` already runs on every sink, keep it that way.
- Windows, Linux, and WSL2 all work: pathlib everywhere, no shell-isms in library code.
- Prose convention for all human-facing markdown: no em dashes (the audit enforces this on the tracked doc set).

## Adding a New Pipeline Stage

1. Add the stage to `Stage` in `state.py` in execution order.
2. Write the stage function in `pipeline/` taking injected components, returning pydantic models.
3. Wire it into the orchestrator with `start_stage` / `complete_stage` checkpoints.
4. If it calls a model backend: type against the `Provider` protocol (never a concrete client), pass `stage=` for the ledger; the orchestrator picks the backend per `engine.mode`.
5. If it touches the GPU: register a `ModelSpec`, access only via `arbiter.acquire`.
6. Add unit tests against fixtures; update `quarry inspect` expectations.
7. Update the docs/Architecture.md diagram and this map if a new module appeared.

## Session Guidance for Agents

- Fan-out suits module-parallel work with frozen interfaces (the ingest and index layers, report/scripts/docs polish). Orchestration, the arbiter, the provider seam, and the state machine are sequential work; do them in one context.
- Keep subagents on cheaper models. The orchestrating model reviews and judges; it does not type boilerplate.
- Read files, run tests, and grep via delegated agents when context is tight.
- Live runs are slow (an hour for a full local report) and search engines suspend IPs that burst: space live runs out, monitor them from their logs, and prefer `quarry resume` over restarts; checkpoints make interruption cheap.

## Do Not

- No LangChain, LangGraph, or orchestration frameworks.
- No bare `except:`; catch what you mean.
- No `print` outside `cli.py` and `scripts/` (ruff T20 enforces this).
- No network in unit tests (pytest-socket enforces this).
- No new dependency without a line in `DECISIONS.md`.
- No model IDs hardcoded in business logic; they live in config.

## Before This Repo Goes Public

1. `make verify` green on a clean clone.
2. CI green on the default selection.
3. README quickstart re-tested from scratch.
4. No badges pointing at private resources.
5. `make audit` returns go. This is the final gate.
