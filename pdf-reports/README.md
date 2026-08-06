# Sample Reports

These PDFs are sample generations: real, unedited pipeline output, included to show what a finished report looks like. Each one was produced end to end on a single RTX 5060 Mobile laptop (8 GB VRAM) by `uv run quarry research "<topic>"`, and every cited sentence in them survived the VERIFY entailment gate before render. In a real run the PDF lands in `data/reports/` beside its markdown source; the branding, contents page, run charts, references with chunk anchors, and the run-facts appendix are all generated, not hand-made.

| File | Engine | Iterations | API cost | Claims verified | Run id |
| --- | --- | --- | --- | --- | --- |
| [iron-air-batteries-local-engine.pdf](iron-air-batteries-local-engine.pdf) | `local` (default) | 3 | $0.00 | 127 of 127 kept | `98f5948eb366` |
| [sodium-ion-batteries-assisted-engine.pdf](sodium-ion-batteries-assisted-engine.pdf) | `assisted` | 2 | $0.0567 | 124 of 125 kept, 1 dropped | `b2aea99978f7` |

The local report cost nothing because the Qwen3 8B writer, the 4B triage model, and the verification cross-encoder all ran on the local GPU; its ledger records real token counts at zero price. The assisted report's cost is one Haiku 4.5 gap audit plus the polish pass over the locally written draft; its verification chart also shows the entailment gate earning its keep, with one unsupported sentence dropped from the final report. A `premium` (Claude-written) sample exists as markdown in [docs/first-test/ExampleReport.md](../docs/first-test/ExampleReport.md); it predates the PDF stage.

These samples live here, tracked in git, rather than in `data/reports/` because `data/` is the gitignored runtime output directory: real runs write there, the repo never does.

Costs, quality trade-offs, and how the engines differ: see the engine table in the [main README](../README.md), [docs/Architecture.md](../docs/Architecture.md), and the full [docs/RunGuide.md](../docs/RunGuide.md).
