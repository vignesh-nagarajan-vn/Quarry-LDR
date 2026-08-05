"""Download local model weights and the llama.cpp server binary.

Fetches, into the gitignored models/ directory:
  * a llama.cpp release build with CUDA support for this platform (providing
    llama-server), plus the separate cudart bundle when the release ships one;
  * the triage GGUF (config: models.triage_gguf_repo / models.triage_gguf_file);
  * the synth GGUF (config: models.synth_gguf_repo / models.synth_gguf_file),
    used by engine.mode local and assisted. Both GGUFs download by default so
    a later engine switch never requires re-running bootstrap; skip flags
    exist for constrained disks.

The embedder and reranker download themselves into the Hugging Face cache on
first use, so they are not fetched here. Idempotent: already-present files
are skipped. Network-using by nature; never invoked by tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
USER_AGENT = "QuarryLDR-bootstrap (+https://github.com/vignesh-nagarajan-vn/Quarry-LDR)"


def _http_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, dest.open("wb") as fh:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            fh.write(block)


def _pick_assets(assets: list[dict]) -> list[dict]:
    """Choose the server build (and on Windows the matching cudart runtime).

    Release asset naming, e.g. b10242:
      llama-b10242-bin-win-cuda-12.4-x64.zip     <- the server build we want
      cudart-llama-bin-win-cuda-12.4-x64.zip     <- separate CUDA runtime DLLs
    CUDA 12.x is preferred over 13.x: 12.x runs on any driver from the 12.8
    generation up, while 13.x needs a newer driver.
    """

    def named(predicate: object) -> list[dict]:
        return [a for a in assets if a["name"].endswith(".zip") and predicate(a["name"].lower())]  # type: ignore[operator]

    if sys.platform == "win32":
        builds = named(lambda n: n.startswith("llama-") and "bin-win-cuda" in n and "x64" in n)
        builds.sort(key=lambda a: ("cuda-12" not in a["name"].lower(), a["name"]))
        cudart = named(lambda n: n.startswith("cudart") and "cuda-12" in n)
        return [*builds[:1], *cudart[:1]]
    builds = named(lambda n: n.startswith("llama-") and "bin-ubuntu" in n and "x64" in n)
    return builds[:1]


def download_llama_cpp(models_dir: Path) -> int:
    target_dir = models_dir / "llama.cpp"
    server_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    if any(target_dir.rglob(server_name)):
        print(f"llama-server already present under {target_dir}, skipping")
        return 0

    print("querying latest llama.cpp release...")
    release = _http_json(GITHUB_API)
    tag = release.get("tag_name", "unknown")
    assets = release.get("assets", [])
    wanted = _pick_assets(assets)
    if not wanted or not wanted[0]["name"].startswith("llama-"):
        names = ", ".join(sorted(a["name"] for a in assets)[:20])
        print(f"ERROR: no llama-server build asset found in release {tag}. Assets: {names}")
        print("fix:  download a llama.cpp build with llama-server manually into models/llama.cpp/")
        return 1

    for asset in wanted:
        zip_path = target_dir / asset["name"]
        print(f"downloading {asset['name']} ({asset['size'] / 1e6:.0f} MB) from release {tag}")
        _download(asset["browser_download_url"], zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)
        zip_path.unlink()
    found = list(target_dir.rglob(server_name))
    if not found:
        print(f"ERROR: extracted release {tag} but {server_name} was not inside")
        return 1
    print(f"llama-server ready: {found[0]}")
    return 0


def download_gguf(models_dir: Path, repo: str, filename: str) -> int:
    target = models_dir / "gguf" / filename
    if target.is_file():
        print(f"GGUF already present: {target}, skipping")
        return 0
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("fix:  uv sync --extra gpu")
        return 1
    print(f"downloading {repo}/{filename} (multi-GB, one time)...")
    target.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=target.parent,
    )
    if not target.is_file():
        found = sorted(target.parent.rglob(filename))
        if found:
            found[0].replace(target)
    print(f"GGUF ready: {target}")
    return 0


def main() -> int:
    from quarry_ldr.config import load_config

    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=cfg.run.models_dir)
    parser.add_argument("--skip-llama", action="store_true")
    parser.add_argument("--skip-gguf", action="store_true", help="skip the triage GGUF")
    parser.add_argument("--skip-synth-gguf", action="store_true", help="skip the synth GGUF")
    args = parser.parse_args()

    exit_code = 0
    if not args.skip_llama:
        exit_code |= download_llama_cpp(args.models_dir)
    if not args.skip_gguf:
        exit_code |= download_gguf(
            args.models_dir, cfg.models.triage_gguf_repo, cfg.models.triage_gguf_file
        )
    if not args.skip_synth_gguf:
        exit_code |= download_gguf(
            args.models_dir, cfg.models.synth_gguf_repo, cfg.models.synth_gguf_file
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
