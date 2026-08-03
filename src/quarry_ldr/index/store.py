"""LanceDB vector store: idempotent adds, ANN search, schema-version checks.

The store's methods are synchronous (LanceDB's Python API is sync); pipeline
code calls them via ``asyncio.to_thread``.

Implemented in M5.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from pydantic import BaseModel

from quarry_ldr.index.schema import (
    SCHEMA_VERSION,
    TABLE_NAME,
    SchemaMismatchError,
    chunk_arrow_schema,
)
from quarry_ldr.ingest.chunk import Chunk


class RetrievedChunk(BaseModel):
    chunk: Chunk
    distance: float


def _schema_fields_match(found: pa.Schema, expected: pa.Schema) -> bool:
    """True iff both schemas declare the same field names, in the same order,
    with the same Arrow types."""
    if list(found.names) != list(expected.names):
        return False
    return all(found.field(name).type.equals(expected.field(name).type) for name in expected.names)


def _detect_found_schema_version(table: Any, found_schema: pa.Schema) -> int:
    """Best-effort read of the ``schema_version`` recorded in an existing,
    non-matching table. Falls back to -1 when it cannot be determined (field
    absent, table empty, or the read itself fails)."""
    if "schema_version" not in found_schema.names:
        return -1
    try:
        head = table.head(1)
        if head.num_rows == 0:
            return -1
        value = head.column("schema_version")[0].as_py()
        return int(value)
    except Exception:
        return -1


class VectorStore:
    """One LanceDB table of chunks + vectors for a run's evidence."""

    def __init__(self, db_path: Path, dim: int) -> None:
        self.db_path = db_path
        self.dim = dim
        self._db: lancedb.DBConnection | None = None
        self._table: Any = None

    def open(self) -> None:
        """Connect and create-or-validate the table (SchemaMismatchError on drift)."""
        db = lancedb.connect(str(self.db_path))
        self._db = db
        expected_schema = chunk_arrow_schema(self.dim)
        if TABLE_NAME in db.list_tables().tables:
            table = db.open_table(TABLE_NAME)
            found_schema = table.schema
            if not _schema_fields_match(found_schema, expected_schema):
                found_version = _detect_found_schema_version(table, found_schema)
                raise SchemaMismatchError(found_version, SCHEMA_VERSION, str(self.db_path))
            self._table = table
        else:
            self._table = db.create_table(TABLE_NAME, schema=expected_schema)

    def _table_or_raise(self) -> Any:
        if self._table is None:
            raise RuntimeError("VectorStore.open() must be called before use")
        return self._table

    def add(self, chunks: Sequence[Chunk], embeddings: NDArray[np.float32]) -> int:
        """Insert chunks with vectors; chunk_ids already present are skipped.
        Returns the number actually inserted."""
        table = self._table_or_raise()
        if not chunks:
            return 0
        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "url": chunk.url,
                "doc_title": chunk.doc_title,
                "heading_path": json.dumps(chunk.heading_path),
                "text": chunk.text,
                "token_count": chunk.token_count,
                "position": chunk.position,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "schema_version": SCHEMA_VERSION,
                "vector": np.asarray(vector, dtype=np.float32).tolist(),
            }
            for chunk, vector in zip(chunks, embeddings, strict=True)
        ]
        result = table.merge_insert("chunk_id").when_not_matched_insert_all().execute(rows)
        return int(result.num_inserted_rows)

    def search(self, query_vec: NDArray[np.float32], top_k: int) -> list[RetrievedChunk]:
        """ANN search, nearest first."""
        table = self._table_or_raise()
        query = np.asarray(query_vec, dtype=np.float32).tolist()
        rows = table.search(query).limit(top_k).to_list()
        results: list[RetrievedChunk] = []
        for row in rows:
            chunk = Chunk(
                chunk_id=row["chunk_id"],
                url=row["url"],
                doc_title=row["doc_title"],
                heading_path=json.loads(row["heading_path"]),
                text=row["text"],
                token_count=row["token_count"],
                position=row["position"],
                start_char=row["start_char"],
                end_char=row["end_char"],
            )
            results.append(RetrievedChunk(chunk=chunk, distance=float(row["_distance"])))
        return results

    def count(self) -> int:
        return self._table_or_raise().count_rows()

    def has(self, chunk_id: str) -> bool:
        table = self._table_or_raise()
        escaped = chunk_id.replace("'", "''")
        return table.count_rows(f"chunk_id = '{escaped}'") > 0
