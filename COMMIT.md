# COMMIT.md

The commit contract for this repo. History is written for a repo that will be public: no personal detail, no internal paths, no secrets, ever.

## Format

Conventional Commits, enforced type set:

`feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`, `build`, `ci`

Scope is a module path segment: `feat(gpu): ...`, `fix(ingest): ...`, `test(pipeline): ...`, `chore(repo): ...` for cross-cutting.

Subject: imperative mood, lowercase, no trailing period, 72 characters maximum.

Body: required for anything touching GPU memory, API cost, or the citation path. State measured impact, not intent. "Reduces synthesis cost from $3.00 to $0.87 on the 10-section fixture run" beats "improves caching".

Footer: `Refs: M<n>` tying the commit to its milestone.

## Granularity

One logical change per commit. A milestone is one commit unless it exceeds roughly 400 changed lines, then split along module boundaries. Parallel milestones commit once per agent's module.

## Pre-commit checklist

1. Tests green (`make test`).
2. `make verify` clean.
3. Secret scan clean (pre-commit runs detect-secrets).
4. Docs updated if behavior changed.
5. `DECISIONS.md` updated if a dependency or design choice changed.

## Worked examples

Good:

```
feat(gpu): enforce hard VRAM budget with LRU eviction in arbiter

Budget violations are now impossible: acquiring a third model whose
footprint would exceed 6656 MB evicts the least recently used resident
first. Measured on the fake backend: peak declared residency 6200 MB
across the embed->rerank->triage swap sequence, previously unbounded.

Refs: M2
```

Why it is good: type and scope match the module, subject is imperative and specific, body states measured impact, footer ties to the milestone.

Bad:

```
Fixed some GPU stuff and updated docs (WIP)
```

Why it is bad: no type or scope, past tense, vague "stuff", bundles docs with code, "WIP" means it should not have been committed, no milestone reference, no measured impact for a GPU-memory change.
