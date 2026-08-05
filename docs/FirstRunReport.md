# First live runs: costs, bugs, and what the money bought

Quarry-LDR was code-complete, 92 percent covered, and green on every mock-based gate before it had ever produced a report. This page documents the first day of live validation (2026-08-04, on the RTX 5060 Mobile target laptop): every run, every dollar, every bug that only real infrastructure could surface, and the measured economics that replaced the design projections.

## The runs

Nine runs spent $9.98 in API calls. Two produced reports; the other seven each bought a specific, permanent fix.

| # | Run | Outcome | API cost | What it taught |
| --- | --- | --- | --- | --- |
| 1 | smoke, first attempt | 400 on the first API call ever | $0.00 | The claude 5 models reject the `temperature` parameter; the provider sent it on every call. Mocked tests could never see this. |
| 2 | smoke | died at search | $0.09 | Cold SearXNG plus an upstream rate limit pushed one query past the 15 s client timeout. |
| 3 | smoke | interrupted (end of session) | $0.09 | Nothing; parked overnight. |
| 4 | smoke | cost cap exceeded at $3.37 | $3.37 | Nothing between triage and synthesis enforced the ~60K-token corpus design. 463 chunks (~300K tokens) went into one cache write. The ledger cap bounds loops, not one oversized call. |
| 5 | smoke | cost cap exceeded at $2.09 | $2.09 | The heuristic token counter undercounts by 1.32x against the API tokenizer, and adaptive thinking truncated the plan call at exactly its max_tokens, paying a retry. |
| 6 | smoke | preflight false negative | $0.00 | Docker's Windows port proxy answers the first request after idle in ~3 s; the 2 s health probe read a healthy SearXNG as down. |
| 7 | smoke | **PASS** | $1.36 | The measured baseline: 1 iteration, 6 sections, 64,145-token cached corpus, citations resolvable. |
| 8 | research, first attempt | killed at zero evidence | $0.10 | Unbounded 60-query search bursts got the IP suspended by upstream engines (a CAPTCHA suspension of 3600 s among them); searches returned empty 200s while the run kept spending. |
| 9 | research, second attempt | **PASS** (interrupted and resumed on purpose) | $2.88 | The substantive baseline, and a live proof of checkpoint resume. |

## The bugs only production could find

Every fix landed with tests, a gate run, and a measured commit body.

| Bug | Fix commit |
| --- | --- |
| Provider sent `temperature`, removed in the claude 5 API generation | `0255442` |
| No evidence budget between triage and synthesis | `b0d4a76` |
| Token budget counted heuristic tokens, 1.32x under the API tokenizer | `8b859d3` |
| Plan call truncated by adaptive thinking inside max_tokens | `8b859d3` |
| Section calls truncated the same way; one section came back empty | `3a44f39` |
| Gap call truncated the same way | `bcfef7a` |
| Search fired all queries at once and got the IP engine-suspended | `e77d24c` |
| SearXNG health probe timeout too tight for Docker's cold port proxy | `bb15a05` |

Two more had surfaced during GPU bring-up the day before: the CLI preflight test was not hermetic against a populated `.env` (`20f62f5`), and gpu-marked tests needed `HF_HUB_OFFLINE` because a warm model cache still triggers a network etag check that the no-network test policy rightly blocks (`5d8dcf4`).

The recurring theme: three of the eight are the same bug in different clothes. Adaptive thinking on the claude 5 models spends inside `max_tokens`, so every call site sized for text-only output truncated in production. If you budget output tokens for a thinking model, budget for the thinking too.

## Anatomy of the substantive report

Topic: whether a dealer's quoting policy and the market response it induces can be learned jointly as a self-consistent fixed point in OTC corporate bond markets. Three search iterations, 15 sub-questions, 404 triaged evidence chunks trimmed to 79 by the token budget, a 57,815-token cached corpus, 10 sections. The report itself is committed at [ExampleReport.md](ExampleReport.md).

| Stage | Model | Calls | Cost |
| --- | --- | --- | --- |
| plan | claude-opus-5 | 1 | $0.13 |
| gap | claude-sonnet-5 | 3 | $0.12 |
| synthesize, cache write | claude-opus-5 | 1 | $0.66 |
| synthesize, cached reads | claude-opus-5 | 9 | $1.98 |
| **total** | | 14 | **$2.88** |

Where the money is: synthesis output. The corpus is paid for once ($0.58 of the write call is the 57,815-token cache write) and then reread nine times at $0.029 per read. Nearly everything else is Opus writing sections, at 5,600 to 8,192 output tokens each with thinking included.

## The caching claim, measured

The design projected that ten section calls over a ~60K-token corpus would cost $0.87 in corpus transport with the 1 hour cache versus $3.00 without. Measured: **$0.84 cached versus $2.89 uncached**. The projection held almost exactly. What moved was the per-report total: the original $1.00 to $1.50 projection predates the claude 5 generation, whose default adaptive thinking roughly doubles output tokens per call. Measured totals are $1.36 for the single-iteration smoke scope and $2.88 for the full three-iteration report, still 4x to 5x under the naive all-API approach.

## The resume exercise

Run 9 was deliberately killed mid-triage in iteration two. `quarry inspect` showed 21 completed stage checkpoints with their persisted payloads and the incomplete triage stage marked running. `quarry resume` replayed all 21 stages in 49 milliseconds, added zero ledger rows, restarted llama-server, re-ran only the interrupted stage, and carried the run to completion. Interruption cost: nothing but wall-clock time.

## The local GPU held up

All three models shared the 8 GB card under the arbiter's 6,656 MB budget with a peak declared residency of 6,654 MB. Sustained triage held 1.0 to 1.9 s per verdict across 35-minute blocks with no thermal degradation at the 70 W power cap. The full findings, including the one-time PTX JIT cost of running a cuda-12.4 llama.cpp build on an sm_120 card, are in the deployment environment section of `DECISIONS.md`.

## Takeaways

1. **A mocked test suite proves plumbing, not physics.** Seven of the eight production bugs were about prices, tokenizers, rate limits, and API parameter drift; none were reachable by respx.
2. **Budget enforcement must sit where the money is created.** The cost cap reads usage blocks after a call returns, so the corpus budget in front of synthesis is what actually bounds the largest single spend.
3. **Thinking models change output economics.** Plan, gap, and section calls all needed roughly doubled headroom, and per-report cost projections written before adaptive thinking needed a 2x correction.
4. **The compression thesis works.** Three iterations of scraping became a 57,815-token corpus, and the API never saw the raw web.
5. **Free infrastructure is rate-limited infrastructure.** Local SearXNG costs nothing per query, but upstream engines suspend bursty IPs; politeness belongs in the search path just as much as in fetch.
6. **Checkpoint everything.** The resume design paid for itself the first day, recovering an interrupted run for free and making deliberate interruption a safe operational tool.
