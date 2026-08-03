"""redact() scrubs everything secret-shaped; logs carry run_id and stage."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import structlog

from quarry_ldr.logging import get_logger, redact, setup_logging


def test_redacts_anthropic_key() -> None:
    fake = "sk-ant-api03-abcdef1234567890"  # pragma: allowlist secret (not-a-real-key)
    assert "sk-ant-REDACTED" in redact(f"key is {fake}")
    assert "abcdef" not in redact(f"key is {fake}")


def test_redacts_header_style_secrets() -> None:
    out = redact("x-api-key: supersecretvalue123")
    assert "supersecretvalue123" not in out
    out = redact('{"api_key": "abc123xyz"}')
    assert "abc123xyz" not in out
    out = redact("Authorization: Bearer eyJhbGciOi.payload.sig")
    assert "eyJhbGciOi" not in out


def test_redacts_url_query_values_keeps_keys() -> None:
    out = redact("https://example.com/search?q=private+question&apikey=zzz9")
    assert "private" not in out
    assert "zzz9" not in out
    assert "?q=" in out and "&apikey=" in out
    assert "https://example.com/search" in out


def test_redact_leaves_plain_text_alone() -> None:
    text = "fetched 40 documents from 12 domains in 8.2s"
    assert redact(text) == text


def test_log_file_is_jsonl_with_context(tmp_path: Path) -> None:
    setup_logging(log_dir=tmp_path, run_id="testrun01")
    structlog.contextvars.bind_contextvars(stage="fetch")
    try:
        get_logger().info("fetch_done", url="https://example.com/?token=abc123secret")
        logging.shutdown()
        log_file = tmp_path / "testrun01.jsonl"
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        event = json.loads(lines[-1])
        assert event["event"] == "fetch_done"
        assert event["run_id"] == "testrun01"
        assert event["stage"] == "fetch"
        assert "abc123secret" not in json.dumps(event)
    finally:
        structlog.contextvars.clear_contextvars()
        setup_logging()  # reset handlers away from tmp_path


def test_setup_logging_is_idempotent(tmp_path: Path) -> None:
    setup_logging(log_dir=tmp_path, run_id="dup")
    setup_logging(log_dir=tmp_path, run_id="dup")
    root = logging.getLogger()
    # one console + one file handler, not stacked duplicates
    assert len(root.handlers) == 2
    setup_logging()
