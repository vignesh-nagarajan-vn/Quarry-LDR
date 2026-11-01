# Update 1 Master Prompt: Report Quality

You are working in Quarry-LDR, a public repo that turns a research topic into a cited report. Read `CLAUDE.md` first: its invariants, commit contract, and conventions bind all of this work. This file is a local working document and is never committed.

## Mission

Make the reports trustworthy and readable on all three engines, without clutter. Update 1 changes the existing pipeline only. The app and setup work is Update 2 and is out of scope.

Definition of done: all six changes are implemented and tested, the default test selection passes, the four live benchmark runs have completed, and the owner can read before and after reports with a scorecard beside them. The owner then assesses and iteration continues from that assessment.

## Owner decisions, already made

| Topic | Decision |
| --- | --- |
| Uncited factual sentences | Cite it or remove it. Try to attach a supporting citation; remove the sentence if nothing supports it. Linking sentences and labeled gap notes stay. |
| Report length | 3,000 to 6,000 words on every engine. Premium keeps its denser writing. |
| Live testing | All three engines, with a $5.00 total API cap across all runs. |
| Git | Local branch `update-1-report-quality`, one logical change per commit, no push, no merge, no tag, no version bump. |

## Measured starting point

From the `v1.0.1` samples and the saved state of run `98f5948eb366`:

- Search found 134 domains, including ScienceDirect, Nature, energy.gov, and formenergy.com, but the report rests mostly on weak pages.
- PDFs are rejected at fetch (`_ALLOWED_CONTENT_TYPES`). 24 fetches ended `non_html` and 95 ended `http_error`.
- One site, `eureka.patsnap.com`, supplied 1,013 of 1,574 evidence items (64 percent), because the rerank top 40 has no diversity control.
- Form Energy, the best-known iron-air company, appears in 281 evidence chunks but only 13 extracted claims, and zero times in the report. Triage claims strip names and the writer is never asked for them.
- Conflicts ship silently: "above 1,000 Wh/kg" beside "300-350 Wh/kg", and 46.25 percent growth beside 58 percent.
- About 13 to 27 percent of longer sentences carry no citation, and VERIFY passes them through.
- VERIFY sees only the first 1,500 characters of each cited chunk, and its reranker measures relevance more than truth. It kept 118/118 and 127/127 on live runs.
- Titles are cut mid-phrase, sub-question text leaks into prose, grouped sections sit under a heading naming only their first question, and the Overview is one wall of text.
- Premium produced about 20,000 words and three malformed markers such as `[2 (sq02)]` that the citation validator cannot see. An invented `[n]` in premium fails at RENDER after the money is spent, and resume replays the same failure.

## Non-negotiable rules

1. The model justifies, the code decides. If a model returns something absurd, a schema or a code check stops it, not a prompt instruction.
2. Every model output is schema-validated with bounded retries, except free-text section prose, which passes code guards.
3. All GPU access goes through the arbiter. Stages stay batched by model. The synth 8B stays resident only alone.
4. Citation numbering must stay deterministic, and resume must reproduce it exactly. The existing resume test must keep passing and be extended if selection changes.
5. Premium's corpus prefix must stay byte stable across section calls.
6. Local work stays at $0 and needs no API key. New checks run locally on every engine.
7. Every stage still checkpoints. Old run payloads must still load: new fields get defaults.
8. Defaults live in both `config/default.yaml` and `config.py`. Model ids stay in config. Each new dependency gets a `DECISIONS.md` line.
9. No new framework, no speculative options, no feature without a test. A new module must earn its place.
10. Treat fetched text and model output as untrusted data. PDF parsing must not be able to crash a run.
11. Unit tests stay CPU-only and offline, with no API key.

## The six changes

Build them in this order. Change 6 comes first so that every later change is measured.

### Change 6: Scorecard

- `src/quarry_ldr/report/scorecard.py`: pure functions from report markdown to a pydantic scorecard. No model calls, no network.
- `quarry score <run_id_or_path>` prints it. `--against <other>` shows a before and after table. Exit 1 when a gated metric misses its threshold.
- Metrics: words, sections, citations, distinct domains, top-1 and top-3 domain share, share of citations by source tier, uncited factual share, malformed markers, cut-off titles, question titles, longest paragraph, repeated sentences, specifics per 1,000 words, required sections present, cost, minutes.
- `evals/thresholds.json` holds versioned thresholds. Threshold changes go in their own commit with a reason.
- `evals/benchmark.yaml` lists the benchmark topics and engines.
- Done when the scorecard runs on all existing reports and the baseline is saved under `data/`.

### Change 1: Rate sources and widen the pool

- `src/quarry_ldr/ingest/sources.py`: deterministic `rate_url()` returning a tier. Tiers: `primary` (government, standards bodies, intergovernmental, official organization sites of the subject), `scholarly` (journals, preprints, university), `established` (reference works, major news and trade press), `general` (default), `commercial` (market-report mills and machine-generated vendor pages). Rules and lists are editable in config. Unknown domains are `general`, never punished.
- Fetch and extract PDFs, using one small, permissively licensed library. Enforce a separate size limit. A bad PDF becomes a rejected document, never a crash. Offsets must still resolve to real text.
- Add one science-category search per sub-question, through the existing concurrency limit.
- Rerank: keep the raw score for the record, and rank per sub-question by score plus a tier adjustment with a per-domain limit, so one site cannot fill the top 40.
- `select_evidence`: add a per-domain share cap with default 0.20, effective share `max(cap, 1 / n_domains)`, so thin corpora still work. Same policy on every path, including the resume rebuild.
- Do not re-triage a (sub-question, chunk) pair already judged in an earlier iteration, if the saved runs confirm this waste. Persist what was judged.
- Report: a short "Sources" summary with tier mix and any concentration warning, in markdown and PDF.
- Acceptance: fixture tests prove the cap, tier order, determinism, thin-corpus behavior, PDF success and failure, and that a flooding domain cannot dominate.

### Change 2: Conflict check

- New checkpointed stage, between the last GAP and SYNTHESIZE, on the local 4B in every engine mode.
- Code pre-filters to claims with comparable content per sub-question. The model proposes conflicts as schema-validated JSON. Code accepts one only if both quoted statements appear in their claims, the two chunks differ, and numeric sides really differ beyond a tolerance. Conflicts must be from different sources.
- Writers are told about conflicts relevant to their section. Premium receives them inside the byte-stable corpus.
- Code renders a "Where sources disagree" section from the accepted conflicts, so it cannot be omitted. VERIFY gates it like any other text.
- Acceptance: fixture tests with planted conflicts and planted hallucinated conflicts, resume replay, and a clean report when there are no conflicts (no empty section).

### Change 3: Require specifics

- Triage claims must keep names, places, dates, and figures.
- The plan names what each sub-question must pin down. Old plans without it still load.
- A deterministic specifics counter feeds the gap digest. Code adds bounded follow-up queries when a sub-question has no names or figures, even if the model declares saturation, within `run.max_iterations`.
- Corpus and section prompts require named players, projects, dates, and figures where evidence has them. Selection prefers chunks with specifics on ties.
- Acceptance: a fixture where the name exists only in chunk text reaches the claim and the corpus, and gap follow-ups fire and stay bounded.

### Change 4: Editing pass and report structure

- The plan also returns short section titles and groups sub-questions into sections. Code validates that every sub-question appears exactly once, with a deterministic fallback.
- Report shape: Key findings (a short bottom line plus 5 to 8 cited bullets), body sections, Where sources disagree, What we could not establish, Sources, References. Remove the separate Overview and Conclusions.
- Key findings are written last from the edited body. Code assigns each bullet's confidence label from the number of independent domains and their tiers. Models never assign it.
- "What we could not establish" is built by code from a final GAP pass over all evidence. GAP therefore runs on the final iteration too; state its measured token and dollar impact.
- Deterministic cleanup: normalize marker forms (`[2 (sq02)]`, `[1, 2]`, `[1-3]`), strip markers that do not resolve on every path, strip leaked sub-question text, drop repeated sentences, and split wall paragraphs.
- Model editing: one guarded pass per section for local, the existing guarded polish for assisted, and no extra Opus edit for premium. Any marker change discards that edit.
- Length: `report.target_words_min` and `report.target_words_max` set per-section targets. The total is logged and scored. Prompts refer to "the sources reviewed", never "the corpus".
- PDF stays fail-soft and renders the new structure.
- Acceptance: fixture tests for every cleanup rule and guard, section validation fallback, confidence labels, and the shape in both markdown and PDF.

### Change 5: Harder fact-check

- Deterministic number check: every figure in a cited sentence must appear in its cited evidence after normalization. Failures go to rewrite-or-drop.
- Model check on every cited sentence with the local 4B: schema-validated `supported | partial | unsupported`. Code decides: only `supported` passes as is.
- Show VERIFY the whole cited chunk within a safe context budget.
- Uncited factual sentences: code classifies them. Search the section's evidence with the reranker, confirm the best candidate with the model check, attach the marker when supported, and drop the sentence otherwise. Linking sentences and labeled gap notes stay.
- A section emptied by drops is removed. Record new counts in the summary, manifest, and PDF.
- Re-calibrate and record what the checks do on fixtures and live runs, including the false-reject rate from a manual sample of removed sentences.
- Acceptance: fixture tests for planted wrong numbers, unsupported sentences, attached citations, kept gap notes, empty sections, and malformed model replies. VERIFY must fail closed for a sentence it cannot judge after retries.

## Order of work

1. Create the branch. Save the scorecard baseline on existing reports.
2. Implement Changes 6, 1, 2, 3, 4, 5, each with tests and its own commit under the commit contract, using `Refs: M19` through `Refs: M24`. Run format, lint, mypy, and tests before each commit. Stage explicit paths; never stage `AGENTS.md` or this file.
3. Bring up search and probe health with three spaced queries, all required healthy. Keep the laptop awake and plugged in.
4. Run the benchmark, spaced out, resuming rather than restarting after an interruption:
   - Iron-air, `local`
   - GLP-1 weight-loss drug safety and effectiveness, `local`
   - Sodium-ion, `assisted`, cap $0.50
   - Enhanced geothermal, `premium`, cap $4.50
5. Stop a run if search returns zero URLs, before any paid call fires on an empty digest. Never exceed $5.00 total. Never rerun a paid run just to see if it passes; ask the owner first.
6. Read every produced report. Check the scorecard, sample removed sentences by hand, and fix what the reading finds. Local-only reruns may continue; further spending needs the owner's approval.
7. Update `DECISIONS.md` with measured numbers, plus `docs/Architecture.md`, `docs/RunGuide.md`, `README.md`, `CLAUDE.md`, and `pdf-reports/README.md` only where behavior changed. Do not replace the tracked sample PDFs until the owner has assessed the new reports.
8. Run the complete gates: format, lint, mypy, tests, and `scripts/pre_public_audit.py`.

## Handoff

Report plainly: what changed, the before and after scorecard, cost and time per engine, reports to read with their paths, what is still weak, and what failed or was skipped. Do not call a threshold met that was not measured. Do not lower a threshold to pass it. If a target proves unrealistic, leave it failing and explain why.
