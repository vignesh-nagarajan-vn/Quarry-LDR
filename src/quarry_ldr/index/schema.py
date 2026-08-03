"""LanceDB table schema, explicitly versioned to catch drift.

A table written by a different SCHEMA_VERSION raises SchemaMismatchError with
the fix (delete the index dir or re-run with a fresh data_dir) instead of
failing mysteriously deep inside a query.

Implemented in M5.
"""

from __future__ import annotations

import pyarrow as pa

SCHEMA_VERSION = 1
TABLE_NAME = "chunks"


class SchemaMismatchError(Exception):
    def __init__(self, found: int, expected: int, path: str) -> None:
        super().__init__(
            f"LanceDB table at {path} has schema v{found}, this build expects "
            f"v{expected}. Delete that directory (it is a rebuildable cache) or "
            "point run.data_dir somewhere fresh."
        )


def chunk_arrow_schema(dim: int) -> pa.Schema:
    """Arrow schema for the chunks table. heading_path is JSON-encoded."""
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("url", pa.string()),
            pa.field("doc_title", pa.string()),
            pa.field("heading_path", pa.string()),
            pa.field("text", pa.string()),
            pa.field("token_count", pa.int32()),
            pa.field("position", pa.int32()),
            pa.field("start_char", pa.int32()),
            pa.field("end_char", pa.int32()),
            pa.field("schema_version", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )
