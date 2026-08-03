"""Polite fetcher: robots.txt respected, per-domain rate limit, disk cache.

Never fetches the same URL twice: the cache is content-addressed by the
SHA-256 of the normalized URL, and a cache hit skips network entirely. Bodies
live on disk (not in the run DB) so a crash mid-fetch loses nothing.

Implemented in M4.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import httpx
from pydantic import BaseModel

from quarry_ldr.config import FetchSettings


class FetchStatus(enum.StrEnum):
    OK = "ok"
    ROBOTS_DISALLOWED = "robots_disallowed"
    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    TOO_LARGE = "too_large"
    NON_HTML = "non_html"
    ERROR = "error"


class FetchResult(BaseModel):
    url: str
    final_url: str
    status: FetchStatus
    http_status: int | None = None
    content_path: Path | None = None
    content_type: str = ""
    from_cache: bool = False
    fetched_at: datetime | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status is FetchStatus.OK and self.content_path is not None

    def read_bytes(self) -> bytes:
        """Load the cached body from disk. Raises if status is not OK."""
        raise NotImplementedError


def normalize_url(url: str) -> str:
    """Canonical form used for cache keys and dedup: lowercase scheme/host,
    default ports stripped, fragment stripped, trailing slash normalized."""
    raise NotImplementedError


def cache_key(url: str) -> str:
    """SHA-256 hex of the normalized URL."""
    raise NotImplementedError


class Fetcher:
    """Async fetcher with robots compliance, politeness, and a disk cache.

    * robots.txt per host is fetched once, parsed with protego, and cached;
      unreachable robots means allow, unparseable means allow (standard).
    * Per-domain token bucket at ``per_domain_rps``; global concurrency via
      a semaphore of ``max_concurrency``.
    * Sends the identifying ``user_agent`` from config. No evasion of any kind.
    """

    def __init__(self, settings: FetchSettings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client

    async def fetch(self, url: str) -> FetchResult:
        """Fetch one URL through robots check, rate limit, and cache."""
        raise NotImplementedError

    async def fetch_many(self, urls: Iterable[str]) -> list[FetchResult]:
        """Fetch a batch concurrently. Input order is preserved in the output."""
        raise NotImplementedError

    async def aclose(self) -> None:
        raise NotImplementedError
