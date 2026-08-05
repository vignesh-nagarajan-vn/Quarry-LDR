# Run Guide

Everything needed to run Quarry-LDR, on every engine, with every knob explained. The [Architecture](Architecture.md) doc explains how the pipeline works; this one explains how to drive it.

## What You Need

| Requirement | Detail |
| --- | --- |
| GPU | NVIDIA, compute capability 8.0 or newer, 8 GB VRAM or more recommended. The design target is an RTX 5060 Mobile (Blackwell, `sm_120`), which needs a CUDA 12.8 capable driver; the project pins the cu128 PyTorch wheel index. |
| Docker | Docker Desktop (Windows/macOS) or Docker Engine (Linux), for the local SearXNG search container. Everything except live search works without it. |
| Python | 3.12. `uv` manages the environment; the bootstrap script installs `uv` itself if missing. |
| Disk | Roughly 12 GB for local models: the llama-server build, the Qwen3 4B and 8B GGUFs, and the Hugging Face cache for the embedder and reranker. |
| API key | Only for the `assisted` and `premium` engines. The default `local` engine makes zero API calls and never needs one. |

## Setup, Once

```bash
git clone https://github.com/vignesh-nagarajan-vn/Quarry-LDR
cd Quarry-LDR
make bootstrap
```

On a fresh Windows machine without GNU make, the same thing is:

```bash
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
```

Bootstrap installs `uv` if needed, syncs dependencies (with the GPU extra when an NVIDIA card is present), and installs the pre-commit hooks. Then:

```bash
cp .env.example .env          # optional: only if you will use assisted/premium
make searxng                  # start SearXNG in Docker (quarry searxng up)
uv run python scripts/download_models.py
uv run quarry verify          # preflight with exact remediation for anything missing
```

`download_models.py` fetches the llama-server release build (with its CUDA runtime bundle on Windows), the triage GGUF (Qwen3 4B), and the synth GGUF (Qwen3 8B). Both GGUFs download by default so a later engine switch never requires re-running setup; `--skip-llama`, `--skip-gguf`, and `--skip-synth-gguf` opt out for constrained disks. The embedder and reranker land in the Hugging Face cache on first GPU use (or run `scripts/bench_vram.py` once to prefetch them).

`quarry verify` is engine-aware: under the default `local` engine a missing API key prints as a skip line, not a failure, and the synth GGUF is required unless the engine is `premium`.

## Running a Report

```bash
uv run quarry research "your topic, phrased the way you want the cover to read"
```

That is the whole interface. The topic string becomes the report title verbatim, so capitalize it the way you want it typeset. Flags:

| Flag | Does |
| --- | --- |
| `--engine local\|assisted\|premium` | Override `engine.mode` for this run |
| `--max-cost <usd>` | Override `run.cost_cap_usd`, the hard API spend cap |
| `--max-iterations <n>` | Override `run.max_iterations`, the search-loop depth |
| `--config <path>` | Layer a user YAML over the defaults |
| `--verbose` / `-v` | Console logging at debug level |

The report and its PDF land in `data/reports/` as `report-<run_id>.md` and `report-<run_id>.pdf`, with the cost ledger and run manifest appended to the markdown and a charts-and-facts appendix inside the PDF. Every run is checkpointed to SQLite as it goes.

## The Three Engines

`engine.mode` decides who serves PLAN, GAP, and SYNTHESIZE. Everything else, including the VERIFY entailment gate, is identical across engines and always local.

### `local` (the default)

```bash
uv run quarry research "your topic"
```

Zero API calls, no key needed, $0.00 enforced by the ledger (local calls are metered with real token counts at zero price under `local/` model ids). The Qwen3 8B writes the plan and every report section from per-section evidence slices with grammar-constrained output; the 4B handles evidence triage and gap checks while already resident, so the loop never pays a model swap. Measured on the design-target laptop: a full three-iteration report takes roughly an hour, most of it the per-chunk triage passes; the 8B writes all sections in two to three minutes at 35.3 tokens per second.

### `assisted`

```bash
uv run quarry research "your topic" --engine assisted --max-cost 0.50
```

Local plan and draft, with Haiku 4.5 doing the gap audits and one polish pass over the assembled draft. The polish is guarded: the citation-marker multiset must survive exactly and the section delimiters must round-trip, or the polish is discarded and the local draft stands. Measured cost of the one-iteration smoke rehearsal: $0.0224, all of it the single polish call; a full-depth run adds one or two small gap calls and stays well under $0.10.

### `premium`

```bash
uv run quarry research "your topic" --engine premium
```

The v0 hybrid behavior, preserved: Opus 5 plans and writes over one prompt-cached, token-budgeted evidence corpus, Sonnet 5 audits coverage each iteration. Measured $1.36 to $2.88 per report on the live validation runs; [first-test/FirstRunReport.md](first-test/FirstRunReport.md) breaks down where every cent goes. The default $5.00 cost cap applies; the ledger raises `CostCapExceeded` mid-run if crossed, and the run stays resumable.

## Configuration

Precedence, lowest to highest: pydantic defaults, `config/default.yaml`, your `--config` YAML (or `QUARRY_CONFIG` env var), then `QUARRY_*` environment variables. Nesting uses `__`:

```bash
QUARRY_ENGINE__MODE=assisted QUARRY_RUN__COST_CAP_USD=0.25 uv run quarry research "topic"
```

Every key, its default, and what moving it does: the configuration reference in [Architecture.md](Architecture.md#configuration-reference). The knobs that matter most in practice:

| Key | Default | When to move it |
| --- | --- | --- |
| `engine.mode` | `local` | Set-and-forget engine choice; `--engine` overrides per run |
| `run.max_iterations` | 3 | Depth vs time: each loop adds search, fetch, and a triage pass |
| `run.cost_cap_usd` | 5.0 | Hard ceiling for the paid engines |
| `report.max_sections` | 12 | Report breadth |
| `report.pdf` | true | Disable the PDF entirely if you only want markdown |
| `verify.floor` | -8.0 | Raise toward 0 for stricter reports, lower to keep more borderline sentences |
| `gpu.vram_budget_mb` | 6656 | Set to ~80 percent of your card's VRAM on non-default hardware |
| `synth.request_timeout_s` | 300 | Raise if a heavily throttled card still times out writing sections |

## Watching, Inspecting, Resuming

```bash
uv run quarry runs                    # list known runs, newest first
uv run quarry inspect <run_id>        # stage-by-stage state as JSON
uv run quarry resume <run_id>         # continue from the last completed stage
```

Every stage transition is a SQLite row, so an interrupted run (crash, power loss, cost cap) resumes by replaying completed stages from their persisted payloads and re-executing only unfinished work. Resume uses the current process config including `engine.mode`: a run resumed under a different engine continues under the new engine from that point, by design.

`quarry searxng up | down | status` manages the search container.

## Rehearsals Before Real Topics

```bash
make smoke-local     # $0 end-to-end rehearsal on the local engine, no key needed
make smoke           # the same rehearsal on the configured engine, $2.00 hard cap
```

Both run the real pipeline (real GPU, real search, real API when the engine calls one) on a trivial fixed topic, print the ledger, and fail if any citation does not resolve. `scripts/smoke.py` also takes `--engine` and `--max-cost` directly, which is how the assisted tier was cost-measured:

```bash
uv run python scripts/smoke.py --engine assisted --max-cost 0.50
```

## Operational Cautions, Learned Live

- **Space out repeated runs.** Upstream search engines have suspended an IP after too many queries in a short window; `search.max_concurrency` bounds the per-run burst, but volume across runs adds up. Probe with one query before spending on a paid run.
- **Keep the machine awake** during a run. Laptops that sleep mid-run resume fine (that is what checkpoints are for), but on some Windows builds Docker needs a reboot after sleep before search works again.
- **Thermal throttling is real.** Sustained runs generate well below burst benchmarks; if section writing times out on your card, raise `synth.request_timeout_s` and resume the run.
- **Different GPU?** Set `gpu.vram_budget_mb` to about 80 percent of VRAM and run `scripts/bench_vram.py` to measure real footprints; the arbiter corrects declared values against measurements, within a plausibility guard.

Symptoms and exact fixes for everything above: [Troubleshooting.md](Troubleshooting.md).
