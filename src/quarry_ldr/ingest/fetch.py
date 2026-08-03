"""Polite fetcher: robots.txt respected, per-domain rate limit, disk cache.

Never fetches the same URL twice: the cache is content-addressed by the
SHA-256 of the normalized URL, and a cache hit skips network entirely. Bodies
live on disk (not in the run DB) so a crash mid-fetch loses nothing.

Implemented in M4.
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from protego import Protego
from pydantic import BaseModel

from quarry_ldr.config import FetchSettings
from quarry_ldr.logging import get_logger

logger = get_logger(component="fetch")

_ALLOWED_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "text/plain"}
_DEFAULT_PORTS = {"http": 80, "https": 443}


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
        if self.status is not FetchStatus.OK or self.content_path is None:
            raise ValueError(f"cannot read bytes for non-ok fetch result (status={self.status})")
        return self.content_path.read_bytes()


def normalize_url(url: str) -> str:
    """Canonical form used for cache keys and dedup: lowercase scheme/host,
    default ports stripped, fragment stripped, trailing slash normalized."""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()

    netloc = hostname
    if parts.username:
        userinfo = parts.username
        if parts.password:
            userinfo = f"{userinfo}:{parts.password}"
        netloc = f"{userinfo}@{netloc}"

    port = parts.port
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        netloc = f"{netloc}:{port}"

    path = parts.path if parts.path else "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def cache_key(url: str) -> str:
    """SHA-256 hex of the normalized URL."""
    return hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()


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
        self._owns_client = client is None
        self._robots_cache: dict[str, Protego | None] = {}
        self._robots_locks: dict[str, asyncio.Lock] = {}
        self._domain_locks: dict[str, asyncio.Lock] = {}
        self._domain_last_request: dict[str, float] = {}
        self._domain_min_interval: dict[str, float] = {}
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(follow_redirects=True)
        return self._client

    def _cache_paths(self, key: str) -> tuple[Path, Path]:
        subdir = self.settings.cache_dir / key[:2]
        return subdir / f"{key}.body", subdir / f"{key}.meta.json"

    def _read_cache(self, url: str, key: str) -> FetchResult | None:
        body_path, meta_path = self._cache_paths(key)
        if not body_path.is_file() or not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cache_meta_unreadable", key=key, error=str(exc))
            return None
        fetched_at_raw = meta.get("fetched_at")
        return FetchResult(
            url=url,
            final_url=meta.get("final_url", url),
            status=FetchStatus.OK,
            http_status=meta.get("http_status"),
            content_path=body_path,
            content_type=meta.get("content_type", ""),
            from_cache=True,
            fetched_at=datetime.fromisoformat(fetched_at_raw) if fetched_at_raw else None,
        )

    def _write_cache(self, key: str, content: bytes, meta: dict[str, Any]) -> Path:
        body_path, meta_path = self._cache_paths(key)
        body_path.parent.mkdir(parents=True, exist_ok=True)
        body_path.write_bytes(content)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return body_path

    def _min_interval_for(self, host: str) -> float:
        base = 1.0 / self.settings.per_domain_rps if self.settings.per_domain_rps > 0 else 0.0
        return max(base, self._domain_min_interval.get(host, 0.0))

    async def _respect_rate_limit(self, host: str) -> None:
        lock = self._domain_locks.setdefault(host, asyncio.Lock())
        async with lock:
            interval = self._min_interval_for(host)
            last = self._domain_last_request.get(host)
            if last is not None and interval > 0:
                wait = interval - (time.monotonic() - last)
                if wait > 0:
                    await asyncio.sleep(wait)
            self._domain_last_request[host] = time.monotonic()

    async def _fetch_robots(self, scheme: str, host: str) -> Protego | None:
        robots_url = f"{scheme}://{host}/robots.txt"
        client = await self._get_client()
        try:
            response = await client.get(
                robots_url,
                headers={"User-Agent": self.settings.user_agent},
                timeout=self.settings.timeout_s,
            )
        except httpx.HTTPError as exc:
            logger.info("robots_fetch_failed", host=host, error=str(exc))
            return None
        if response.status_code >= 400:
            logger.info("robots_not_found", host=host, status=response.status_code)
            return None
        try:
            return Protego.parse(response.text)
        except ValueError as exc:
            logger.info("robots_unparseable", host=host, error=str(exc))
            return None

    async def _check_robots(self, scheme: str, host: str, url: str) -> bool:
        if not host:
            return True
        lock = self._robots_locks.setdefault(host, asyncio.Lock())
        async with lock:
            if host not in self._robots_cache:
                self._robots_cache[host] = await self._fetch_robots(scheme, host)
        parser = self._robots_cache[host]
        if parser is None:
            return True
        allowed = parser.can_fetch(url, self.settings.user_agent)
        if allowed:
            delay = parser.crawl_delay(self.settings.user_agent)
            if delay is not None and delay > self._min_interval_for(host):
                self._domain_min_interval[host] = delay
        return bool(allowed)

    async def _do_fetch(self, url: str, key: str) -> FetchResult:
        client = await self._get_client()
        headers = {"User-Agent": self.settings.user_agent}
        try:
            async with client.stream(
                "GET", url, headers=headers, timeout=self.settings.timeout_s
            ) as response:
                final_url = str(response.url)
                content_type = (
                    response.headers.get("content-type", "").split(";")[0].strip().lower()
                )

                content_length = response.headers.get("content-length")
                if content_length is not None:
                    try:
                        too_large = int(content_length) > self.settings.max_bytes
                    except ValueError:
                        too_large = False
                    if too_large:
                        return FetchResult(
                            url=url,
                            final_url=final_url,
                            status=FetchStatus.TOO_LARGE,
                            http_status=response.status_code,
                            content_type=content_type,
                        )

                if response.status_code >= 400:
                    return FetchResult(
                        url=url,
                        final_url=final_url,
                        status=FetchStatus.HTTP_ERROR,
                        http_status=response.status_code,
                        content_type=content_type,
                    )

                if content_type not in _ALLOWED_CONTENT_TYPES:
                    return FetchResult(
                        url=url,
                        final_url=final_url,
                        status=FetchStatus.NON_HTML,
                        http_status=response.status_code,
                        content_type=content_type,
                    )

                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > self.settings.max_bytes:
                        return FetchResult(
                            url=url,
                            final_url=final_url,
                            status=FetchStatus.TOO_LARGE,
                            http_status=response.status_code,
                            content_type=content_type,
                        )
                content = bytes(chunks)
                status_code = response.status_code
        except httpx.TimeoutException as exc:
            return FetchResult(url=url, final_url=url, status=FetchStatus.TIMEOUT, error=str(exc))
        except httpx.HTTPError as exc:
            return FetchResult(url=url, final_url=url, status=FetchStatus.ERROR, error=str(exc))

        fetched_at = datetime.now(UTC)
        body_path = self._write_cache(
            key,
            content,
            {
                "url": url,
                "final_url": final_url,
                "http_status": status_code,
                "content_type": content_type,
                "fetched_at": fetched_at.isoformat(),
            },
        )
        return FetchResult(
            url=url,
            final_url=final_url,
            status=FetchStatus.OK,
            http_status=status_code,
            content_path=body_path,
            content_type=content_type,
            from_cache=False,
            fetched_at=fetched_at,
        )

    async def fetch(self, url: str) -> FetchResult:
        """Fetch one URL through robots check, rate limit, and cache."""
        key = cache_key(url)
        cached = self._read_cache(url, key)
        if cached is not None:
            return cached

        normalized = normalize_url(url)
        parsed = urlsplit(normalized)
        host = parsed.hostname or ""
        scheme = parsed.scheme or "https"

        allowed = await self._check_robots(scheme, host, normalized)
        if not allowed:
            logger.info("robots_disallowed", url=url, host=host)
            return FetchResult(url=url, final_url=url, status=FetchStatus.ROBOTS_DISALLOWED)

        async with self._semaphore:
            await self._respect_rate_limit(host)
            return await self._do_fetch(url, key)

    async def fetch_many(self, urls: Iterable[str]) -> list[FetchResult]:
        """Fetch a batch concurrently. Input order is preserved in the output."""
        url_list = list(urls)
        results = await asyncio.gather(*(self.fetch(u) for u in url_list))
        return list(results)

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
