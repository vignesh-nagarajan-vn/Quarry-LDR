# HANDOFF

2026-08-05. Branch main at c2b44c2, working tree clean, 11 commits since
v0.9.0-beta. This file is a session snapshot: delete it once the Next list
below is empty.

## Where Things Stand

v1 (the local-first rebuild, plan: `~/.claude/plans/abstract-soaring-wren.md`)
is code-complete through M17 of M18. `engine.mode` (local | assisted |
premium) routes PLAN/GAP/SYNTHESIZE; local is the default, runs with no API
key, and ledgers real token counts at $0 under `local/` model ids. VERIFY
scores every cited sentence against its cited chunks and rewrites or drops
failures on the triage 4B. RENDER ships a branded Typst PDF beside the
markdown. Gate at HEAD: ruff, mypy, 351 tests passing, 93.7 percent
coverage. The v0 hybrid is archived on `archive/v0-hybrid-api` and released
as v0.9.0-beta.

## In Flight

A live $0 local run (run_id `1dfcc750a2e5`, sand-battery topic) is
executing on this laptop via `uv run quarry resume 1dfcc750a2e5`. It has
already validated local PLAN, the search loop, and the gap-digest fix live;
it had not reached SYNTHESIZE/VERIFY/RENDER when this session paused. Its
process holds `.venv/Scripts/quarry.exe`, which blocks plain `uv run` from
syncing (matplotlib and typst are locked in pyproject but not yet installed
in the venv). First command after it exits: `uv sync`.

## Next (delete items as they land)

1. `uv sync`, then inspect `data/reports/report-1dfcc750a2e5.md`. The
   resumed process predates M15+ code, so run one fresh
   `uv run quarry research "<topic>"` to see VERIFY and the PDF live.
2. Calibrate verify.floor on this GPU:
   `uv run python -m pytest -m gpu tests/integration/test_verify_calibration.py -s --no-cov`;
   put the measured floor in config.py and default.yaml and the measured
   numbers in DECISIONS.md (M15 follow-up commit).
3. Assisted live smoke, only with the user's explicit go-ahead (hard cap
   $0.50): `uv run quarry research "<topic>" --engine assisted --max-cost 0.5`;
   record the measured cost in DECISIONS.md and the M16 entry.
4. M18 per the plan. Already landed early (commit cccfac8): README Title
   Case, engine table, VERIFY/PDF figure, status note, and the License and
   Acknowledgements section crediting local-deep-research. Still to do:
   fill the README's "to be measured" assisted cost and any final numbers,
   Architecture/Troubleshooting/CLAUDE.md updates, `scripts/smoke.py
   --engine` + a `smoke-local` Makefile target, pyproject version 1.0.0rc1,
   then `make verify` and `make audit`.
5. Re-test the README quickstart with no API key set; update `quarry
   inspect` expectations if stage output changed.

## Gotchas

- This machine is the RTX 5060 Mobile laptop, not the 4060 desktop.
- App Control blocks venv exe shims: use `uv run python -m pytest` and
  `python -m mypy`, never bare `pytest`/`mypy`.
- While any quarry process runs, commit with `UV_NO_SYNC=1` (pre-commit
  hooks call `uv run`) and prefer `uv run --no-sync` generally.
- verify.floor ships at 0.0 placeholder; bge cross-encoder scores are
  logits, so 0.0 already gates. Do not ship v1 without calibration.
- Search engines suspended this IP once before (DECISIONS, M10): do not
  burst repeated live runs back to back.

## Read These First

1. `~/.claude/plans/abstract-soaring-wren.md` (the approved v1 plan)
2. DECISIONS.md, section "v1 engine tiers (local-first rebuild, M11+)"
3. CLAUDE.md (invariants; M18 is where its map and commands get updated)
