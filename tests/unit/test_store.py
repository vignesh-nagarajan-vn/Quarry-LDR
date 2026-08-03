"""VectorStore over a real LanceDB on tmp_path: round trip, idempotency, ANN
recall sanity, and schema-drift detection. No GPU, no network."""

from __future__ import annotations

from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
import pytest
from numpy.typing import NDArray

from quarry_ldr.index.schema import (
    SCHEMA_VERSION,
    TABLE_NAME,
    SchemaMismatchError,
)
from quarry_ldr.index.store import RetrievedChunk, VectorStore
from quarry_ldr.ingest.chunk import Chunk

DIM = 8


def _chunk(chunk_id: str, position: int, heading_path: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        url="https://example.example/doc",
        doc_title="Example Doc",
        heading_path=heading_path if heading_path is not None else ["Intro", "Background"],
        text=f"text for {chunk_id}",
        token_count=10,
        position=position,
        start_char=position * 100,
        end_char=position * 100 + 50,
    )


def _basis_vector(index: int, dim: int = DIM) -> NDArray[np.float32]:
    """A near-basis unit vector: mostly at ``index`` with small noise elsewhere,
    so hand-built vectors are trivially distinguishable by nearest-neighbor."""
    vec = np.full(dim, 0.01, dtype=np.float32)
    vec[index % dim] = 1.0
    vec = vec / np.linalg.norm(vec)
    return vec.astype(np.float32)


def test_open_creates_empty_table(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "db", dim=DIM)
    store.open()
    assert store.count() == 0


def test_add_and_search_round_trip_known_nearest(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "db", dim=DIM)
    store.open()
    chunks = [_chunk("c0", 0), _chunk("c1", 1), _chunk("c2", 2)]
    embeddings = np.stack([_basis_vector(0), _basis_vector(3), _basis_vector(6)])
    inserted = store.add(chunks, embeddings)
    assert inserted == 3

    results = store.search(_basis_vector(0), top_k=3)
    assert len(results) == 3
    assert results[0].chunk.chunk_id == "c0"
    assert results[0].distance <= results[1].distance <= results[2].distance
    assert results[0].distance == pytest.approx(0.0, abs=1e-4)


def test_add_is_idempotent(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "db", dim=DIM)
    store.open()
    chunks = [_chunk("a", 0), _chunk("b", 1)]
    embeddings = np.stack([_basis_vector(0), _basis_vector(1)])
    first = store.add(chunks, embeddings)
    second = store.add(chunks, embeddings)
    assert first == 2
    assert second == 0
    assert store.count() == 2


def test_add_skips_only_already_present_ids(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "db", dim=DIM)
    store.open()
    store.add([_chunk("a", 0)], np.stack([_basis_vector(0)]))
    inserted = store.add(
        [_chunk("a", 0), _chunk("b", 1)],
        np.stack([_basis_vector(0), _basis_vector(1)]),
    )
    assert inserted == 1
    assert store.count() == 2


def test_has_true_and_false(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "db", dim=DIM)
    store.open()
    store.add([_chunk("present", 0)], np.stack([_basis_vector(0)]))
    assert store.has("present") is True
    assert store.has("absent") is False


def test_heading_path_round_trip(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "db", dim=DIM)
    store.open()
    heading = ["Chapter 1", "Section A", "Subsection i"]
    store.add([_chunk("h", 0, heading_path=heading)], np.stack([_basis_vector(0)]))
    results = store.search(_basis_vector(0), top_k=1)
    assert results[0].chunk.heading_path == heading


def test_count_matches_number_added(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "db", dim=DIM)
    store.open()
    chunks = [_chunk(f"c{i}", i) for i in range(5)]
    embeddings = np.stack([_basis_vector(i) for i in range(5)])
    store.add(chunks, embeddings)
    assert store.count() == 5


def test_ann_recall_sanity(tmp_path: Path) -> None:
    """200 random normalized vectors plus one known target: querying near the
    target must return it first, i.e. nearest-neighbor search is not broken."""
    rng = np.random.default_rng(42)
    dim = 16
    store = VectorStore(tmp_path / "db", dim=dim)
    store.open()

    n_random = 200
    random_vectors = rng.normal(size=(n_random, dim)).astype(np.float32)
    random_vectors /= np.linalg.norm(random_vectors, axis=1, keepdims=True)
    random_chunks = [_chunk(f"rand{i}", i) for i in range(n_random)]

    target_vector = rng.normal(size=dim).astype(np.float32)
    target_vector /= np.linalg.norm(target_vector)
    target_chunk = _chunk("target", n_random)

    all_chunks = [*random_chunks, target_chunk]
    all_vectors = np.vstack([random_vectors, target_vector[None, :]]).astype(np.float32)
    store.add(all_chunks, all_vectors)

    results = store.search(target_vector, top_k=5)
    assert results[0].chunk.chunk_id == "target"
    assert results[0].distance == pytest.approx(0.0, abs=1e-4)


def test_schema_drift_raises_schema_mismatch_error(tmp_path: Path) -> None:
    db_path = tmp_path / "db"
    bad_schema = pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), DIM)),
        ]
    )
    db = lancedb.connect(str(db_path))
    db.create_table(TABLE_NAME, schema=bad_schema)

    store = VectorStore(db_path, dim=DIM)
    with pytest.raises(SchemaMismatchError) as exc_info:
        store.open()
    message = str(exc_info.value)
    assert "v-1" in message  # schema_version column absent: undeterminable
    assert f"v{SCHEMA_VERSION}" in message
    assert str(db_path) in message


def test_reopen_existing_valid_table_succeeds(tmp_path: Path) -> None:
    db_path = tmp_path / "db"
    store1 = VectorStore(db_path, dim=DIM)
    store1.open()
    store1.add([_chunk("x", 0)], np.stack([_basis_vector(0)]))

    store2 = VectorStore(db_path, dim=DIM)
    store2.open()
    assert store2.count() == 1
    assert store2.has("x") is True


def test_retrieved_chunk_model_fields(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "db", dim=DIM)
    store.open()
    store.add([_chunk("m", 0)], np.stack([_basis_vector(0)]))
    results = store.search(_basis_vector(0), top_k=1)
    assert isinstance(results[0], RetrievedChunk)
    assert isinstance(results[0].chunk, Chunk)
    assert isinstance(results[0].distance, float)
