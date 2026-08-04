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
- **ASYNC230/240 ignored in ruff**: small local file reads/writes in async functions are deliberate; heavy IO (LanceDB, model loads) already routes through `asyncio.to_thread`, so the rule has nothing left to catch and only adds noise.
- **Audit exempts `.env` from the working-tree secret scan and redacts matched secrets (M10)**: a configured machine always holds the real key in `.env`, which made `make audit` unpassable exactly where reports are produced; safety holds because the separate `.env`-is-gitignored check still fails the audit on its own, and matched secret content is now reported by location only, never echoed (the previous behavior printed the live key to the console).
- **`HF_HUB_OFFLINE=1` set in `tests/conftest.py` (M10)**: with a warm cache, huggingface_hub still makes a per-file etag check over the network at model load, which pytest-socket rightly blocks; offline mode makes gpu-marked tests deterministic on any machine whose cache already holds the models (fetched outside pytest, e.g. by `scripts/bench_vram.py`).
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
- Token counting is pluggable (`TokenCounter` protocol); both tests and the runtime pipeline currently use the ~4 chars/token heuristic. An HF-tokenizer counter would change only chunk granularity, not correctness, and would drag tokenizer downloads into the chunk stage; revisit if measured chunk sizes drift badly from 512 tokens.
- **Resume is replay, not special-case code**: every stage checkpoint stores the payload the next stage needs; re-driving a run returns persisted payloads for completed stages instantly (identical timestamps prove no recomputation) and executes only unfinished stages. Embeddings persist as npz per iteration since LanceDB only holds post-dedup rows.
- Dedup: 64-bit SimHash over 5-word shingles, Hamming <= 3, OR embedding cosine >= 0.92. Thresholds validated against the fixture near-duplicate set in M5.
- Fetch politeness: 1 request/second/domain token bucket, 8 global concurrent fetches, 20s timeout, 5 MB response cap, identifying user agent.
- Embedder default `BAAI/bge-m3` (dense, dim 1024); `Qwen/Qwen3-Embedding-0.6B` stays a config swap.
- Triage model `Qwen3-4B-Instruct-2507` Q4_K_M GGUF via llama-server, 8K context to keep KV cache small. Sourced from `unsloth/Qwen3-4B-Instruct-2507-GGUF`: Qwen publishes no official GGUF for the 2507 instruct variant (verified against the HF API, 2026-08-03); unsloth's conversion carries the expected filename and chat template.
- SearXNG on host port 8888 and llama-server on 8555: both defaults dodge the crowded 8080.
- Declared VRAM footprints are measured values since M10 (embedder 2186 MB, reranker 2128 MB, triage 3600 MB declared-only; see the M10 entry below); the arbiter still corrects declared values against `torch.cuda.mem_get_info` at every load.
- **SimHash Hamming <= 3 is nearly toothless standalone (measured, M5).** On the fixture corpus at ~100-200 words/chunk, scattered synonym swaps push near-duplicate chunk pairs to Hamming 0-14, so the shipped default of 3 catches under half of them SimHash-only, while the closest genuinely-unrelated pair sits at 13. The corpus regression test therefore runs SimHash-only at Hamming 10 (>=80 percent catch rate, zero false positives); the shipped default stays 3 because production dedup pairs SimHash with embedding cosine >= 0.92, where SimHash only needs to be a high-precision complement.
- **lancedb `list_tables()` over deprecated `table_names()`** (lancedb 0.36); merge_insert on chunk_id gives idempotent adds natively.
- **Windows WDDM hides child-process VRAM from mem_get_info (measured, M6).** llama-server's ~3.5 GB allocation measured as 0 MB from the parent process, which would have "corrected" the triage footprint to zero and broken all budget math. The arbiter now treats any measurement under 25 percent of the declared footprint as implausible and keeps the declared value (logged as footprint_measurement_implausible). Live lifecycle numbers on the RTX 4060: llama-server (Qwen3-4B-Instruct-2507 Q4_K_M, 8K context, full offload) healthy in 4.8 s, one structured triage verdict in 0.94 s, clean terminate under the arbiter.
- **llama.cpp release assets need a cudart-aware matcher.** The Windows CUDA runtime ships as a separate cudart-*.zip whose name also matches naive bin-win-cuda-x64 filters; download_models.py picks the llama-* build first and prefers CUDA 12.x (driver 572.16 supports 12.8, not 13.x). The server must be spawned with absolute -m paths because cwd is set to the binary directory for DLL resolution.
- **Synthesis corpus budget enforced at report.corpus_budget_tokens (M10, measured):** the first live smoke run shipped all 463 triaged chunks (about 300K tokens) into synthesis because nothing between triage and the corpus builder enforced the design's ~60K payload; the 1h cache write alone cost about $3.28 and the run recorded $3.37 against a $0.50 cap (the ledger cap is checked from usage blocks after each call returns, so it bounds loops, not a single oversized call). Evidence is now trimmed before corpus build, round-robin across sub-questions in descending rerank order, so coverage degrades evenly instead of dropping whole sub-questions.
- **Corpus budget calibrated to 45000 heuristic tokens and plan max_tokens raised to 8192 (M10, measured):** the second live run measured the heuristic-to-API tokenizer ratio at 1.32x (60000 heuristic tokens became a 79456-token cache write), so the default budget is 45000 heuristic tokens to land the design's ~60K API-token corpus; the ratio is corpus-shaped and worth re-measuring if chunking changes. The same run also truncated its first plan call at exactly max_tokens=4096 (claude-opus-5 thinks inside the output budget by default) and paid a $0.10 retry; plan headroom is now 8192. Smoke additionally caps report.max_sections at 6, mirroring its single-iteration depth override: the rehearsal exercises cache write, reads, and citations without paying for full report breadth.
- **Search concurrency bounded at search.max_concurrency, default 4, and the SearXNG per-engine timeout raised to 6 s (M10, measured):** unbounded search_many gathers fired 56 to 60 concurrent queries per run; after five runs in one morning the shared engines suspended the IP (startpage CAPTCHA with suspended_time=3600, another engine 429 at 180 s, duckduckgo connect timeouts), and searches returned zero URLs while the pipeline kept spending on plan calls. Fetch always had politeness machinery; search now gets a semaphore. The 6 s engine timeout stops throttled-but-alive engines from being scored as dead at SearXNG's 3 s default.
- **VRAM footprints measured on the deployment RTX 5060 Laptop GPU (M10, 2026-08-03):** embedder (bge-m3) 2186 MB and reranker (bge-reranker-v2-m3) 2128 MB as `mem_get_info` deltas under the arbiter, against the 1400/1300 placeholders; config defaults now carry the measured values. Triage stays declared 3600 MB: WDDM reported 0 MB for the llama-server child and the implausibility guard (CLAUDE.md invariant 8) correctly kept the declared value, logging `footprint_measurement_implausible`. Throughput on this machine: 95.7 texts/s embed, 111.3 pairs/s rerank, one triage verdict in 1.71 s. Peak observed usage across the embed, rerank, and triage swap sequence was 5553 MB against the 6656 MB budget, with one LRU eviction (embedder) when triage joined the reranker.
- **Reranker regression measured on the RTX 4060 (M5):** mean nDCG@6 on the 5-query hand-labeled set is 0.958 for bge-reranker-v2-m3 versus 0.938 for a lexical word-overlap baseline. The margin is small because synthetic fixture chunks share exact wording with their queries, which flatters the baseline; the gpu-marked test asserts model >= baseline so a broken reranker path still fails. torch 2.11.0+cu128 verified with an on-device matmul (scripts/verify_gpu.py PASS).
