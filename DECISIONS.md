# DECISIONS.md

Every deviation from the build spec and every delegated design decision, with a one line rationale. Measured numbers replace assumptions as they land.

## Environment corrections (measured 2026-08-03)

- **GPU is an RTX 4060 (Ada, compute capability 8.9, 8188 MiB, driver 572.16), not an RTX 5060 Blackwell.** `nvidia-smi` measured. `verify_gpu.py` therefore checks capability >= (8, 0) plus a real on-device matmul instead of hard-requiring `sm_120`; the cu128 wheel pin stays, so Blackwell cards also work. The 8 GB VRAM budget logic is unchanged.
- **Host is native Windows 11; WSL is not installed.** Runtime targets Windows natively (torch cu128 Windows wheels, llama.cpp Windows CUDA binary); code stays cross-platform via pathlib and platform detection; CI runs Ubuntu so the Linux path stays honest.
- **Docker is not installed, not merely stopped.** SearXNG ships fully configured; `quarry searxng up` detects missing Docker and prints exact remediation. Nothing else depends on Docker.
- **Tooling installed during M0:** uv 0.12.1 (astral.sh installer), GNU make 4.4.1 (winget, ezwinports), Python 3.12.13 (uv-managed). System Python was 3.9.

## Tooling and repo

- **ruff for both lint and format** (black-compatible defaults): one tool, one config, no second formatter dependency.
- **mypy for type checking** with `disallow_untyped_defs`: strict enough to catch real bugs without fighting untyped third-party libs.
- **pytest-socket** blocks all non-loopback sockets in the default selection: "no network in unit tests" is enforced, not hoped.
- **detect-secrets** for secret scanning: pip-installable, baseline committed, runs in pre-commit as a local hook so hooks need no network.
- **Local pre-commit hooks** (`language: system`) running `uv run ...`: hook versions are exactly the project's pinned versions.
- **Coverage excludes `raise NotImplementedError` lines**: standard exclusion that keeps the 80 percent gate honest through the interfaces-first build (stubs vanish as milestones land).
- **`bootstrap.ps1` added alongside `bootstrap.sh`**: the dev machine is native Windows; spec listed only the sh script.
- **`quarry searxng up|down|status` CLI subcommands** instead of an extra script: the spec's script list stays as specified, and the Docker preflight lives with the rest of the CLI UX.
- **T20 (flake8-print) ruff rule** enforces "no print outside the CLI layer" mechanically.

## Dependencies (each new dep gets a line here)

- **typer**: CLI with completion and testing support, minimal boilerplate.
- **pydantic + pydantic-settings**: boundary models everywhere; layered env config.
- **structlog + rich**: JSON file logs plus readable console.
- **httpx**: async HTTP for fetcher, SearXNG client, and llama-server client.
- **trafilatura**: boilerplate removal that preserves heading structure via XML output.
- **protego**: robots.txt parsing with correct wildcard handling; stdlib robotparser mishandles wildcards.
- **anthropic**: official SDK; transport-level mocking in tests.
- **aiosqlite**: async SQLite for the run store; WAL mode for crash safety.
- **lancedb + pyarrow**: embedded ANN store, no server process, versioned schema.
- **numpy**: vectors and cosine math.
- **tenacity**: retry/backoff for the API client.
- **pytest, pytest-asyncio, pytest-cov, pytest-socket, respx**: test stack; respx mocks httpx at the transport layer.
- **torch (cu128 index), sentence-transformers, transformers**: GPU extra only, never pulled by CI.

## Pipeline parameters (from spec unless noted; tuned values get measured justification)

- Chunking 512 target / 64 overlap tokens; fragments under 48 tokens merge backward.
- Token counting is pluggable (`TokenCounter` protocol): runtime uses the embedder's HF tokenizer, tests use a ~4 chars/token heuristic so the suite needs no downloads.
- Dedup: 64-bit SimHash over 5-word shingles, Hamming <= 3, OR embedding cosine >= 0.92. Thresholds validated against the fixture near-duplicate set in M5.
- Fetch politeness: 1 request/second/domain token bucket, 8 global concurrent fetches, 20s timeout, 5 MB response cap, identifying user agent.
- Embedder default `BAAI/bge-m3` (dense, dim 1024); `Qwen/Qwen3-Embedding-0.6B` stays a config swap.
- Triage model `Qwen3-4B-Instruct-2507` Q4_K_M GGUF via llama-server, 8K context to keep KV cache small.
- SearXNG on host port 8888 and llama-server on 8555: both defaults dodge the crowded 8080.
- Declared VRAM footprints (embedder 1400 MB, reranker 1300 MB, triage 3600 MB) are placeholders until `bench_vram.py` measures them in M10; the arbiter corrects declared values against `torch.cuda.mem_get_info` at every load.
- **SimHash Hamming <= 3 is nearly toothless standalone (measured, M5).** On the fixture corpus at ~100-200 words/chunk, scattered synonym swaps push near-duplicate chunk pairs to Hamming 0-14, so the shipped default of 3 catches under half of them SimHash-only, while the closest genuinely-unrelated pair sits at 13. The corpus regression test therefore runs SimHash-only at Hamming 10 (>=80 percent catch rate, zero false positives); the shipped default stays 3 because production dedup pairs SimHash with embedding cosine >= 0.92, where SimHash only needs to be a high-precision complement.
- **lancedb `list_tables()` over deprecated `table_names()`** (lancedb 0.36); merge_insert on chunk_id gives idempotent adds natively.
