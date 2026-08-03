"""Embedder: batching, normalization, empty-input shortcut, arbiter residency.

CPU tests swap the arbiter's registered loader for a deterministic fake
(the fake-loader pattern: construct the real component so it registers its
spec, then re-register the same name with a fake loader while the model is
not yet resident). The one real-model test is @pytest.mark.gpu and
deselected by default.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.arbiter import BudgetExceededError, ModelSpec, VramArbiter
from quarry_ldr.gpu.embedder import EMBED_DIM_BY_MODEL, Embedder


class FakeSentenceTransformer:
    """Deterministic stand-in for ``SentenceTransformer.encode``.

    Vectors are seeded from a hash of the text (not Python's randomized
    ``hash()``) so the same text always produces the same unit vector.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.encode_calls: list[dict[str, Any]] = []

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        normalize_embeddings: bool | None = None,
        convert_to_numpy: bool | None = None,
    ) -> NDArray[np.float32]:
        self.encode_calls.append(
            {
                "n_texts": len(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "convert_to_numpy": convert_to_numpy,
            }
        )
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=self.dim).astype(np.float32)
            vectors[i] = vec / np.linalg.norm(vec)
        return vectors


def _swap_in_fake(
    embedder: Embedder, arbiter: VramArbiter
) -> tuple[FakeSentenceTransformer, list[int]]:
    """Replace the real loader registered in ``Embedder.__init__`` with a fake
    one, per the required CPU-test pattern (allowed only while not resident)."""
    fake = FakeSentenceTransformer(embedder.dim)
    load_calls: list[int] = []

    def fake_loader() -> FakeSentenceTransformer:
        load_calls.append(1)
        return fake

    arbiter.register(
        ModelSpec(
            name=Embedder.ARBITER_NAME,
            footprint_mb=10,
            loader=fake_loader,
            unloader=lambda m: None,
        )
    )
    return fake, load_calls


def _make_embedder(
    cfg: QuarryConfig,
) -> tuple[Embedder, VramArbiter, FakeSentenceTransformer, list[int]]:
    arbiter = VramArbiter(budget_mb=cfg.gpu.vram_budget_mb)
    embedder = Embedder(cfg, arbiter)
    fake, load_calls = _swap_in_fake(embedder, arbiter)
    return embedder, arbiter, fake, load_calls


async def test_embed_texts_shape_dtype_and_normalization(cfg: QuarryConfig) -> None:
    embedder, _arbiter, _fake, _calls = _make_embedder(cfg)
    vectors = await embedder.embed_texts(["alpha", "beta", "gamma"])
    assert vectors.shape == (3, embedder.dim)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


async def test_embed_texts_batch_size_passthrough(cfg: QuarryConfig) -> None:
    cfg.gpu.embed_batch_size = 7
    embedder, _arbiter, fake, _calls = _make_embedder(cfg)
    await embedder.embed_texts(["a", "b"])
    assert fake.encode_calls[-1]["batch_size"] == 7
    assert fake.encode_calls[-1]["normalize_embeddings"] is True
    assert fake.encode_calls[-1]["convert_to_numpy"] is True


async def test_embed_texts_empty_input_shortcut(cfg: QuarryConfig) -> None:
    embedder, arbiter, fake, load_calls = _make_embedder(cfg)
    vectors = await embedder.embed_texts([])
    assert vectors.shape == (0, embedder.dim)
    assert vectors.dtype == np.float32
    assert load_calls == []  # loader never invoked: the model was never acquired
    assert fake.encode_calls == []
    assert "embedder" not in arbiter.resident_models()


async def test_embed_query_shape(cfg: QuarryConfig) -> None:
    embedder, _arbiter, _fake, _calls = _make_embedder(cfg)
    vector = await embedder.embed_query("hello world")
    assert vector.shape == (embedder.dim,)
    assert vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


async def test_arbiter_residency_after_call(cfg: QuarryConfig) -> None:
    embedder, arbiter, _fake, _calls = _make_embedder(cfg)
    await embedder.embed_texts(["one"])
    assert "embedder" in arbiter.resident_models()


async def test_embeddings_deterministic_for_same_text(cfg: QuarryConfig) -> None:
    embedder, _arbiter, _fake, _calls = _make_embedder(cfg)
    first = await embedder.embed_texts(["repeat me"])
    second = await embedder.embed_texts(["repeat me"])
    assert np.array_equal(first, second)


def test_registration_uses_configured_footprint(cfg: QuarryConfig) -> None:
    cfg.gpu.footprints_mb["embedder"] = 2000
    arbiter = VramArbiter(budget_mb=1999)
    with pytest.raises(BudgetExceededError):
        Embedder(cfg, arbiter)


def test_registration_falls_back_to_default_footprint(cfg: QuarryConfig) -> None:
    cfg.gpu.footprints_mb.pop("embedder", None)
    too_small = VramArbiter(budget_mb=1399)
    with pytest.raises(BudgetExceededError):
        Embedder(cfg, too_small)
    just_right = VramArbiter(budget_mb=1400)
    Embedder(cfg, just_right)  # does not raise: default footprint is 1400 MB


def test_dim_selection_known_and_unknown_models(cfg: QuarryConfig) -> None:
    assert EMBED_DIM_BY_MODEL["BAAI/bge-m3"] == 1024
    cfg.models.embedder = "some/unknown-model"
    arbiter = VramArbiter(budget_mb=cfg.gpu.vram_budget_mb)
    embedder = Embedder(cfg, arbiter)
    assert embedder.dim == 1024


@pytest.mark.gpu
async def test_real_encode_three_sentences(cfg: QuarryConfig) -> None:
    arbiter = VramArbiter(budget_mb=cfg.gpu.vram_budget_mb)
    embedder = Embedder(cfg, arbiter)
    vectors = await embedder.embed_texts(
        ["The quick brown fox.", "Jumps over the lazy dog.", "GPU acceleration is fast."]
    )
    assert vectors.shape == (3, 1024)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
