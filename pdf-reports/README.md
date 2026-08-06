# Sample Reports

These PDFs are sample generations: real, unedited pipeline output, included to show what a finished report looks like on every engine tier. Each one was produced end to end on a single RTX 5060 Mobile laptop (8 GB VRAM) by `uv run quarry research "<topic>"`, and every cited sentence in them survived the VERIFY entailment gate before render. In a real run the PDF lands in `data/reports/` beside its markdown source; the branding, contents page, run charts, references with chunk anchors, and the run-facts appendix are all generated, not hand-made.

| File | Engine | Iterations | API cost | Claims verified | Run id |
| --- | --- | --- | --- | --- | --- |
| [iron-air-batteries-local-engine.pdf](iron-air-batteries-local-engine.pdf) | `local` (default) | 3 | $0.00 | 127 of 127 kept | `98f5948eb366` |
| [sodium-ion-batteries-assisted-engine.pdf](sodium-ion-batteries-assisted-engine.pdf) | `assisted` | 2 | $0.0567 | 124 of 125 kept, 1 dropped | `b2aea99978f7` |
| [heat-pumps-assisted-engine.pdf](heat-pumps-assisted-engine.pdf) | `assisted` | 3 | $0.1188 | 121 of 121 kept | `2d30c621f031` |
| [enhanced-geothermal-premium-engine.pdf](enhanced-geothermal-premium-engine.pdf) | `premium` | 2 | $2.5540 | 372 of 376 kept, 4 dropped | `6bc4d96f41d3` |

Notes worth reading before comparing them:

- The local report cost nothing because the Qwen3 8B writer, the 4B triage model, and the verification cross-encoder all ran on the local GPU; its ledger records real token counts at zero price.
- The sodium-ion assisted report is the polished path: one Haiku 4.5 gap audit plus an applied polish pass, and its verification chart shows the entailment gate earning its keep with one unsupported sentence dropped.
- The heat-pumps assisted report demonstrates the polish guard instead: Haiku's polish changed the citation-marker multiset, so the guard discarded it and the local draft shipped. Its cost also predates a schema fix that has since halved assisted gap spend.
- The premium report is Claude-written (Opus 5 over a prompt-cached corpus, Sonnet 5 gap audit) at roughly three times the citation density of the local engine. It ran during an upstream search throttling window, so it stands on fewer sources than premium normally gathers, and it says so in its own prose. This copy is re-rendered from the run's checkpointed draft with a since-landed heading fix; the content is byte-identical to the run's record in every cited sentence.

These samples live here, tracked in git, rather than in `data/reports/` because `data/` is the gitignored runtime output directory: real runs write there, the repo never does.

Costs, quality trade-offs, and how the engines differ: see the engine table in the [main README](../README.md), [docs/Architecture.md](../docs/Architecture.md), and the full [docs/RunGuide.md](../docs/RunGuide.md).
