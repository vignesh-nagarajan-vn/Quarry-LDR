"""Batched local embedding on the GPU, mediated by the arbiter.

Default model: BAAI/bge-m3, dense vectors, dimension 1024, L2-normalized
float32 output so cosine similarity is a plain dot product downstream.

Implemented in M5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.arbiter import VramArbiter

EMBED_DIM_BY_MODEL: dict[str, int] = {
    "BAAI/bge-m3": 1024,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
}


class Embedder:
    """Embeds text batches under ``arbiter.acquire("embedder")``.

    Registration with the arbiter happens in ``__init__``; the model is only
    actually loaded inside :meth:`embed_texts` / :meth:`embed_query`.
    """

    ARBITER_NAME = "embedder"

    def __init__(self, cfg: QuarryConfig, arbiter: VramArbiter) -> None:
        self.cfg = cfg
        self.arbiter = arbiter
        self.model_id = cfg.models.embedder
        self.dim = EMBED_DIM_BY_MODEL.get(self.model_id, 1024)
        self.batch_size = cfg.gpu.embed_batch_size

    async def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed a batch. Returns (len(texts), dim) float32, L2-normalized."""
        raise NotImplementedError

    async def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a single query string. Returns (dim,) float32, L2-normalized."""
        raise NotImplementedError
