"""SearXNG client: free, unlimited local metasearch over JSON API.

Implemented in M4.
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel

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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = client

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Run one query; results are URL-deduplicated and capped at num_results.

        Raises SearxngError with remediation when the instance is down or
        returns HTML instead of JSON.
        """
        raise NotImplementedError

    async def search_many(self, queries: list[str], num_results: int = 10) -> list[SearchResult]:
        """Run several queries concurrently; merged and URL-deduplicated."""
        raise NotImplementedError

    async def health(self) -> bool:
        """True when the instance answers a trivial JSON query."""
        raise NotImplementedError
