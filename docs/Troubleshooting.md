# Troubleshooting

Symptoms observed while building and running Quarry-LDR, with their causes and exact fixes. `uv run quarry verify` and `scripts/smoke.py` print most of these remediations automatically.

| Symptom | Cause and fix |
| --- | --- |
| `verify_gpu.py` fails on capability | Wheels lack kernels for your GPU. For `sm_120` (Blackwell, the RTX 5060 Mobile design target) install the cu128 index wheels: `uv sync --extra gpu` uses it already; check your driver supports CUDA 12.8. |
| SearXNG returns HTML instead of JSON | The `json` format is not enabled. Confirm `docker/searxng/settings.yml` lists `json` under `search.formats`, then `quarry searxng down && quarry searxng up`. |
| llama-server fails to start, port in use | Another process holds port 8555. Change `triage.port` or free the port. |
| llama-server fails to start with a missing CUDA DLL on Windows | The llama.cpp Windows release ships the CUDA runtime as a separate `cudart-*.zip`, not inside the server build. `scripts/download_models.py` downloads and extracts both the `llama-server` build and the matching cudart bundle; if extraction was interrupted, delete `models/llama.cpp/` and rerun it. |
| `SchemaMismatchError` from LanceDB | The index was written by an older schema. Delete the index directory under `data/`; it is a rebuildable cache. |
| CUDA OOM under the arbiter | Declared footprints are lower than reality on your card. Run `scripts/bench_vram.py` and update `gpu.footprints_mb`, or lower batch sizes. |
| Measured VRAM looks like 0 MB right after llama-server loads | Windows WDDM does not expose a child process's VRAM to `mem_get_info`, so the parent sees no allocation. The arbiter treats any measurement under 25 percent of the declared footprint as implausible and keeps the declared value instead of zeroing the budget. |
| `download_models.py` can't find an official Qwen3-4B-Instruct-2507 GGUF | Qwen publishes no official GGUF for the 2507 instruct variant. The default `models.triage_gguf_repo` points at `unsloth/Qwen3-4B-Instruct-2507-GGUF`, a conversion that carries the expected filename and chat template. |
| API 429 / 529 | The client retries with backoff up to `api.max_retries`. Persistent 429 means your account rate limits; lower concurrency or wait. |
| `make searxng` says Docker is missing | Install Docker Desktop (Windows/macOS) or Docker Engine (Linux) and start it. Everything except live search works without it. |
| Searches suddenly return zero results while SearXNG answers fast | Upstream engines have suspended your IP after too many queries in a short window (a CAPTCHA suspension can last an hour). Wait 30 to 60 minutes and space runs out; `search.max_concurrency` bounds the per-run burst. Probe with one real query before spending on a run. |
| Docker Desktop crashes at startup after the machine slept (Windows) | Some Windows builds corrupt AF_UNIX socket creation after sleep; the corrupt reparse point files cannot be deleted. Reboot, and if a stale file still blocks startup, rename `%LOCALAPPDATA%\Docker\run` aside and relaunch. Keep the machine awake during long runs. |
| `pytest -m gpu` fails with SocketConnectBlockedError on a fresh machine | The embedder and reranker are not in the Hugging Face cache yet, and tests are network-blocked by design. Run `scripts/bench_vram.py` once to fetch them; the test config sets `HF_HUB_OFFLINE` so a warm cache never touches the network. |
| `section_hit_max_tokens` warnings in the synthesis log | The model spent its output budget (thinking included) on that section; the section is kept as written. An empty section is retried once automatically against the cached corpus. |
| `uv python install 3.12` fails with "Missing expected target directory" | A uv-on-this-machine bug seen on one Windows box. Harmless when a system Python 3.12 exists; `uv sync` falls back to it and the venv works. |
