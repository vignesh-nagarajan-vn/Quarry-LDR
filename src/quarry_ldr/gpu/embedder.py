"""Batched local embedding on the GPU, mediated by the arbiter.

Default model: BAAI/bge-m3, dense vectors, dimension 1024, L2-normalized
float32 output so cosine similarity is a plain dot product downstream.

Implemented in M5.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.arbiter import ModelSpec, VramArbiter

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
        arbiter.register(
            ModelSpec(
                name=self.ARBITER_NAME,
                footprint_mb=cfg.gpu.footprints_mb.get("embedder", 1400),
                loader=self._load_model,
                unloader=self._unload_model,
            )
        )

    def _load_model(self) -> Any:
        """Real loader: imported lazily so CPU tests never import torch/sentence_transformers."""
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_id, device="cuda")

    def _unload_model(self, model: Any) -> None:
        """Dereference the model; the arbiter empties the CUDA cache afterward."""
        del model

    async def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed a batch. Returns (len(texts), dim) float32, L2-normalized."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        async with self.arbiter.acquire(self.ARBITER_NAME) as model:
            vectors = await asyncio.to_thread(
                model.encode,
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return np.asarray(vectors, dtype=np.float32)

    async def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a single query string. Returns (dim,) float32, L2-normalized."""
        vectors = await self.embed_texts([text])
        return vectors[0]
