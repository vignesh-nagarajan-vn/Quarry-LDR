"""Unit tests for the SearxClient (SearXNG JSON API client).

All HTTP is mocked with respx; no network is required or permitted.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from quarry_ldr.ingest.search import (
    SEARXNG_DOWN_REMEDIATION,
    SEARXNG_JSON_REMEDIATION,
    SearchResult,
    SearxClient,
    SearxngError,
)

BASE_URL = "http://searxng.example.test"
SEARCH_URL = f"{BASE_URL}/search"


def _json_response(results: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={"results": results})


async def test_search_parses_results_into_search_result_models() -> None:
    with respx.mock:
        respx.get(SEARCH_URL).mock(
            return_value=_json_response(
                [
                    {
                        "url": "https://a.example/1",
                        "title": "Title A",
                        "content": "Snippet A",
                        "engine": "duckduckgo",
                        "score": 1.5,
                    }
                ]
            )
        )
        client = SearxClient(base_url=BASE_URL)
        results = await client.search("heat storage")

    assert results == [
        SearchResult(
            url="https://a.example/1",
            title="Title A",
            snippet="Snippet A",
            engine="duckduckgo",
            score=1.5,
            query="heat storage",
        )
    ]


async def test_search_dedups_by_url_preserving_first_seen_and_caps() -> None:
    payload = [
        {
            "url": "https://a.example/1",
            "title": "First",
            "content": "c1",
            "engine": "e1",
            "score": 1.0,
        },
        {
            "url": "https://b.example/2",
            "title": "Second",
            "content": "c2",
            "engine": "e2",
            "score": 2.0,
        },
        # Duplicate of the first URL: must be dropped, not counted toward cap.
        {
            "url": "https://a.example/1",
            "title": "First dup",
            "content": "d",
            "engine": "e1",
            "score": 9.0,
        },
        {
            "url": "https://c.example/3",
            "title": "Third",
            "content": "c3",
            "engine": "e3",
            "score": 3.0,
        },
        {
            "url": "https://d.example/4",
            "title": "Fourth",
            "content": "c4",
            "engine": "e4",
            "score": 4.0,
        },
    ]
    with respx.mock:
        respx.get(SEARCH_URL).mock(return_value=_json_response(payload))
        client = SearxClient(base_url=BASE_URL)
        results = await client.search("q", num_results=3)

    assert [r.url for r in results] == [
        "https://a.example/1",
        "https://b.example/2",
        "https://c.example/3",
    ]
    # First-seen title/score win, not the later duplicate.
    assert results[0].title == "First"
    assert results[0].score == 1.0


async def test_search_maps_content_to_snippet_and_sets_query() -> None:
    with respx.mock:
        respx.get(SEARCH_URL).mock(
            return_value=_json_response(
                [
                    {
                        "url": "https://a.example/1",
                        "title": "T",
                        "content": "the actual snippet text",
                        "engine": "brave",
                        "score": 0.75,
                    }
                ]
            )
        )
        client = SearxClient(base_url=BASE_URL)
        results = await client.search("my query")

    assert results[0].snippet == "the actual snippet text"
    assert results[0].engine == "brave"
    assert results[0].query == "my query"


async def test_search_empty_results_returns_empty_list() -> None:
    with respx.mock:
        respx.get(SEARCH_URL).mock(return_value=_json_response([]))
        client = SearxClient(base_url=BASE_URL)
        results = await client.search("nothing here")

    assert results == []


async def test_search_html_body_raises_searxng_error_with_json_remediation() -> None:
    with respx.mock:
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text="<html><body>SearXNG web UI</body></html>",
            )
        )
        client = SearxClient(base_url=BASE_URL)
        with pytest.raises(SearxngError) as exc_info:
            await client.search("q")

    assert exc_info.value.remediation == SEARXNG_JSON_REMEDIATION
    assert "settings.yml" in str(exc_info.value)


async def test_search_html_like_body_without_content_type_header_also_raises() -> None:
    """Body starting with '<' triggers the same remediation even if the
    content-type header is missing or wrong."""
    with respx.mock:
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                text="<!doctype html><html></html>",
            )
        )
        client = SearxClient(base_url=BASE_URL)
        with pytest.raises(SearxngError) as exc_info:
            await client.search("q")

    assert exc_info.value.remediation == SEARXNG_JSON_REMEDIATION


async def test_search_403_raises_json_remediation() -> None:
    with respx.mock:
        respx.get(SEARCH_URL).mock(return_value=httpx.Response(403, json={"error": "forbidden"}))
        client = SearxClient(base_url=BASE_URL)
        with pytest.raises(SearxngError) as exc_info:
            await client.search("q")

    assert exc_info.value.remediation == SEARXNG_JSON_REMEDIATION
    assert "settings.yml" in str(exc_info.value)


async def test_search_connect_error_raises_down_remediation() -> None:
    with respx.mock:
        respx.get(SEARCH_URL).mock(side_effect=httpx.ConnectError("connection refused"))
        client = SearxClient(base_url=BASE_URL)
        with pytest.raises(SearxngError) as exc_info:
            await client.search("q")

    assert exc_info.value.remediation == SEARXNG_DOWN_REMEDIATION
    assert "make searxng" in str(exc_info.value)


async def test_search_connect_timeout_raises_down_remediation() -> None:
    with respx.mock:
        respx.get(SEARCH_URL).mock(side_effect=httpx.ConnectTimeout("timed out"))
        client = SearxClient(base_url=BASE_URL)
        with pytest.raises(SearxngError) as exc_info:
            await client.search("q")

    assert exc_info.value.remediation == SEARXNG_DOWN_REMEDIATION


async def test_search_other_http_error_includes_status_code_and_down_remediation() -> None:
    with respx.mock:
        respx.get(SEARCH_URL).mock(return_value=httpx.Response(500, json={"error": "boom"}))
        client = SearxClient(base_url=BASE_URL)
        with pytest.raises(SearxngError) as exc_info:
            await client.search("q")

    assert exc_info.value.remediation == SEARXNG_DOWN_REMEDIATION
    assert "500" in str(exc_info.value)


async def test_search_many_merges_and_dedups_across_queries() -> None:
    with respx.mock:
        respx.get(SEARCH_URL, params={"q": "q1", "format": "json"}).mock(
            return_value=_json_response(
                [
                    {
                        "url": "https://a.example/1",
                        "title": "A1",
                        "content": "",
                        "engine": "e",
                        "score": 1.0,
                    },
                    {
                        "url": "https://shared.example/x",
                        "title": "Shared",
                        "content": "",
                        "engine": "e",
                        "score": 1.0,
                    },
                ]
            )
        )
        respx.get(SEARCH_URL, params={"q": "q2", "format": "json"}).mock(
            return_value=_json_response(
                [
                    # Same URL as in q1's results: must be dropped from the merge.
                    {
                        "url": "https://shared.example/x",
                        "title": "Shared dup",
                        "content": "",
                        "engine": "e",
                        "score": 5.0,
                    },
                    {
                        "url": "https://b.example/2",
                        "title": "B2",
                        "content": "",
                        "engine": "e",
                        "score": 1.0,
                    },
                ]
            )
        )
        client = SearxClient(base_url=BASE_URL)
        results = await client.search_many(["q1", "q2"], num_results=10)

    # Query order first, then result order within each query; dedup keeps first-seen.
    assert [r.url for r in results] == [
        "https://a.example/1",
        "https://shared.example/x",
        "https://b.example/2",
    ]
    assert results[1].title == "Shared"
    assert results[0].query == "q1"
    assert results[2].query == "q2"


async def test_search_many_bounds_concurrency() -> None:
    client = SearxClient(base_url=BASE_URL, max_concurrency=3)
    active = 0
    peak = 0

    async def fake_search(query: str, num_results: int = 10) -> list:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)  # yield so other bounded tasks can interleave
        active -= 1
        return []

    client.search = fake_search  # type: ignore[method-assign]
    await client.search_many([f"q{i}" for i in range(12)], num_results=5)
    assert peak <= 3


async def test_health_true_on_success() -> None:
    with respx.mock:
        respx.get(SEARCH_URL).mock(
            return_value=_json_response(
                [
                    {
                        "url": "https://a.example/1",
                        "title": "T",
                        "content": "",
                        "engine": "e",
                        "score": 1.0,
                    }
                ]
            )
        )
        client = SearxClient(base_url=BASE_URL)
        assert await client.health() is True


async def test_health_false_when_searxng_down() -> None:
    with respx.mock:
        respx.get(SEARCH_URL).mock(side_effect=httpx.ConnectError("refused"))
        client = SearxClient(base_url=BASE_URL)
        assert await client.health() is False


async def test_health_false_on_html_response() -> None:
    with respx.mock:
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200, headers={"content-type": "text/html"}, text="<html></html>"
            )
        )
        client = SearxClient(base_url=BASE_URL)
        assert await client.health() is False


async def test_injected_client_is_reused_and_never_closed() -> None:
    async with httpx.AsyncClient() as injected:
        with respx.mock:
            respx.get(SEARCH_URL).mock(return_value=_json_response([]))
            client = SearxClient(base_url=BASE_URL, client=injected)
            await client.search("q")
            assert injected.is_closed is False
