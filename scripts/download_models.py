"""Download local model weights and the llama.cpp server binary.

Fetches, into the gitignored models/ directory:
  * the embedder and reranker from Hugging Face (via huggingface_hub);
  * the triage GGUF (models.triage_gguf_repo / models.triage_gguf_file);
  * a llama.cpp release build with CUDA support for this platform,
    providing llama-server.

Idempotent: already-present files are skipped. Implemented in M6.
"""

from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError("implemented in milestone M6")


if __name__ == "__main__":
    sys.exit(main())
