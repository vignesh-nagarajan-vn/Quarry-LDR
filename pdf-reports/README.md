# Sample Reports

These PDFs are sample generations: real, unedited pipeline output, included to show what a finished report looks like. Each one was produced end to end on a single RTX 5060 Mobile laptop (8 GB VRAM) by `uv run quarry research "<topic>"`, and every cited sentence in them survived the VERIFY entailment gate before render. In a real run the PDF lands in `data/reports/` beside its markdown source; the branding, contents page, run charts, references with chunk anchors, and the run-facts appendix are all generated, not hand-made.

| File | Engine | Iterations | API cost | Claims verified | Run id |
| --- | --- | --- | --- | --- | --- |
| [iron-air-batteries-local-engine.pdf](iron-air-batteries-local-engine.pdf) | `local` (default) | 3 | $0.00 | 127 of 127 kept | `98f5948eb366` |
| [sand-battery-assisted-engine.pdf](sand-battery-assisted-engine.pdf) | `assisted` | 1 | $0.0224 | 84 of 84 kept | `b41bf5186509` |

The local report cost nothing because the Qwen3 8B writer, the 4B triage model, and the verification cross-encoder all ran on the local GPU; its ledger records real token counts at zero price. The assisted report's entire cost is one Haiku 4.5 polish pass over the locally written draft. A `premium` (Claude-written) sample exists as markdown in [docs/ExampleReport.md](../docs/ExampleReport.md); it predates the PDF stage.

Costs, quality trade-offs, and how the engines differ: see the engine table in the [main README](../README.md) and [docs/Architecture.md](../docs/Architecture.md).
