"""SearXNG client: free, unlimited local metasearch over JSON API.

Implemented in M4.
"""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager, nullcontext

import httpx
from pydantic import BaseModel

from quarry_ldr.logging import get_logger

logger = get_logger()

SEARXNG_JSON_REMEDIATION = (
    "SearXNG returned HTML instead of JSON. The json format is probably not "
    "enabled: check docker/searxng/settings.yml contains 'json' under "
    "search.formats, then restart with `quarry searxng down` and `quarry searxng up`."
)
SEARXNG_DOWN_REMEDIATION = (
    "SearXNG is not reachable. Start it with `make searxng` (requires Docker "
    "Desktop installed and running), or point search.searxng_url at a running instance."
)


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str = ""
    engine: str = ""
    score: float = 0.0
    query: str = ""


class SearxngError(Exception):
    """SearXNG unreachable or misconfigured; carries an exact remediation."""

    def __init__(self, message: str, remediation: str) -> None:
        self.remediation = remediation
        super().__init__(f"{message}\nfix: {remediation}")


class SearxClient:
    """Thin async client over a SearXNG instance's /search JSON endpoint."""

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 15.0,
        client: httpx.AsyncClient | None = None,
        max_concurrency: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = client
        self.max_concurrency = max_concurrency

    def _client_ctx(self) -> AbstractAsyncContextManager[httpx.AsyncClient]:
        """Context manager yielding a usable client.

        Reuses the injected client without closing it, or opens (and closes)
        a temporary one when none was injected.
        """
        if self._client is not None:
            return nullcontext(self._client)
        return httpx.AsyncClient(timeout=self.timeout_s)

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Run one query; results are URL-deduplicated and capped at num_results.

        Raises SearxngError with remediation when the instance is down or
        returns HTML instead of JSON.
        """
        url = f"{self.base_url}/search"
        async with self._client_ctx() as client:
            try:
                response = await client.get(
                    url,
                    params={"q": query, "format": "json"},
                    timeout=self.timeout_s,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                raise SearxngError(
                    f"could not connect to SearXNG at {self.base_url}: {exc}",
                    SEARXNG_DOWN_REMEDIATION,
                ) from exc

            content_type = response.headers.get("content-type", "")
            body = response.text

            if (
                response.status_code == 403
                or "text/html" in content_type
                or body.lstrip().startswith("<")
            ):
                raise SearxngError(
                    f"SearXNG returned HTML/non-JSON (status {response.status_code}) for {url}",
                    SEARXNG_JSON_REMEDIATION,
                )

            if response.status_code >= 400:
                raise SearxngError(
                    f"SearXNG returned HTTP {response.status_code} for {url}",
                    SEARXNG_DOWN_REMEDIATION,
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise SearxngError(
                    f"SearXNG returned invalid JSON for {url}: {exc}",
                    SEARXNG_JSON_REMEDIATION,
                ) from exc

        raw_results = payload.get("results", []) if isinstance(payload, dict) else []
        seen: set[str] = set()
        results: list[SearchResult] = []
        for item in raw_results:
            item_url = item.get("url")
            if not item_url or item_url in seen:
                continue
            seen.add(item_url)
            results.append(
                SearchResult(
                    url=item_url,
                    title=item.get("title", "") or "",
                    snippet=item.get("content", "") or "",
                    engine=item.get("engine", "") or "",
                    score=item.get("score", 0.0) or 0.0,
                    query=query,
                )
            )
            if len(results) >= num_results:
                break
        return results

    async def search_many(self, queries: list[str], num_results: int = 10) -> list[SearchResult]:
        """Run several queries with bounded concurrency; merged, URL-deduplicated.

        The bound keeps a burst of plan queries from fanning out to every
        upstream engine at once: an unbounded 60-query gather got this IP
        CAPTCHA-suspended by multiple engines in live runs.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def bounded_search(query: str) -> list[SearchResult]:
            async with semaphore:
                return await self.search(query, num_results=num_results)

        per_query_results = await asyncio.gather(*(bounded_search(query) for query in queries))
        seen: set[str] = set()
        merged: list[SearchResult] = []
        for query_results in per_query_results:
            for result in query_results:
                if result.url in seen:
                    continue
                seen.add(result.url)
                merged.append(result)
        return merged

    async def health(self) -> bool:
        """True when the instance answers a trivial JSON query."""
        try:
            await self.search("test", num_results=1)
        except SearxngError:
            logger.warning("searxng_health_check_failed", base_url=self.base_url)
            return False
        return True
