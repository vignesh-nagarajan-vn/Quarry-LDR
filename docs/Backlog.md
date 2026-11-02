# Backlog

Open work, as of 2026-11-02. Measured numbers live in [DECISIONS.md](../DECISIONS.md).

## Not yet measured

- Premium benchmark under Update 1 (enhanced geothermal, cap $4.50). Benchmarking stopped before it ran.
- One complete report from live search through `quarry app`.
- Owner assessment of the new reports. Replace the samples in `pdf-reports/` only after it.

## Report quality

- **Word floor:** reports land at 2,380 to 2,914 words against the 3,000 minimum. Find why sections stop short (output limits or VERIFY drops) rather than padding.
- **Dangling references:** when VERIFY drops a sentence, the next one can open on "These findings", "This cost difference", or "For instance" with nothing before it. Rewrite or drop the follower.
- **Repetition:** near-duplicate sentences and per-section "In conclusion" paragraphs pass the `repeated_sentences` check.
- **Source concentration:** GLP-1 top-site share is 0.333 (nih.gov). The site cap limits corpus tokens, not what writers cite.
- **Weak sources:** VERIFY checks support, not truth, so errors in company and blog pages survive (sodium-ion is 69 percent such citations).
- **Disagreements:** merge bullets that share a side. Recall is partial: GLP-1 adherence of 58 to 82 percent against three of four patients stopping within a year went unflagged.
- **Reference numbers skip** now that only cited references are listed. Renumber at render if readers find the gaps confusing.

## Runs and search

- **Search suspensions:** one run of 72 to 78 queries can bring about 20 hours of refusals. A keyed search API option would remove the dependency on scraped engines.
- **Sleep:** a closed lid on battery puts a run in Modern Standby for hours. Warn when a run starts on battery.
- **One process per GPU:** the arbiter guards only its own process, so a terminal run beside an app run can exceed the VRAM budget.

## Downloadable app (planned, not started)

- Phase 1 gate: run bge-m3 and the reranker as GGUF under llama.cpp and match their scores against the current models.
- Then llama.cpp for all four models (no torch, no Docker), a Tauri shell over the existing page, a frozen Python backend, models downloaded on first run, and keyed search by default.
- Owner decisions: the default search provider and key, the code-signing identity, and pricing.

## Release

- `update-2-app` is pushed and not merged. No version bump or tag yet.
