"""Structured logging: JSON lines to file, readable console, redaction by default.

Every event is passed through :func:`redact` before it reaches any sink, so
API keys, bearer tokens, header-style secrets, and URL query parameter values
never land in a log file. Pipeline code binds ``run_id`` and ``stage`` via
``structlog.contextvars`` so every event carries both.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

import structlog

_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    # Anthropic-style keys anywhere in a string.
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"), "sk-ant-REDACTED"),
    # Bearer tokens. Must run before the key:value pattern below, which would
    # otherwise consume the word "Bearer" as the value and leave the token.
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]+=*"), "Bearer REDACTED"),
    # key=value / key: value pairs for common secret-bearing names.
    (
        re.compile(
            r"(?i)\b(api[_-]?key|x-api-key|authorization|access[_-]?token|secret[_-]?key|"
            r"client[_-]?secret|password|token)\b(['\"]?\s*[:=]\s*['\"]?)([^\s'\",;&]+)"
        ),
        r"\1\2REDACTED",
    ),
    # URL query parameter values: keep the key, drop the value.
    (re.compile(r"([?&][\w.%\-]+=)[^&\s\"'<>]+"), r"\1REDACTED"),
]


def redact(text: str) -> str:
    """Scrub secrets and URL parameter values from a string."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def _redact_event(logger: object, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    for key, value in event_dict.items():
        if isinstance(value, str):
            event_dict[key] = redact(value)
    return event_dict


_SHARED_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    _redact_event,
]


def setup_logging(
    log_dir: Path | None = None,
    run_id: str | None = None,
    verbose: bool = False,
) -> None:
    """Configure structlog + stdlib logging with console and optional file sinks.

    Safe to call more than once; handlers are replaced, not stacked.
    """
    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
            foreign_pre_chain=_SHARED_PROCESSORS,
        )
    )
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{run_id}.jsonl" if run_id else "quarry.jsonl"
        file_handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=_SHARED_PROCESSORS,
            )
        )
        root.addHandler(file_handler)

    if run_id is not None:
        structlog.contextvars.bind_contextvars(run_id=run_id)

    # Third-party chatter stays at WARNING unless verbose.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if verbose else logging.WARNING)


def get_logger(**initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, optionally pre-bound with context."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger()
    if initial_values:
        logger = logger.bind(**initial_values)
    return logger
