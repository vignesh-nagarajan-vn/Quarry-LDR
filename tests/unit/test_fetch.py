"""Fetcher: URL normalization, disk cache, robots compliance, guard rails."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from quarry_ldr.config import QuarryConfig
from quarry_ldr.ingest.fetch import (
    Fetcher,
    FetchResult,
    FetchStatus,
    cache_key,
    normalize_url,
)


def _robots_text(fixtures_dir: Path, name: str) -> str:
    return (fixtures_dir / "robots" / f"{name}.txt").read_text(encoding="utf-8")


def _allow_robots(host: str) -> None:
    respx.get(f"https://{host}/robots.txt").mock(return_value=httpx.Response(404))


# --------------------------------------------------------------------------
# normalize_url / cache_key
# --------------------------------------------------------------------------


def test_normalize_url_lowercases_scheme_and_host() -> None:
    assert normalize_url("HTTP://Example.COM/Path") == "http://example.com/Path"


def test_normalize_url_strips_default_ports() -> None:
    assert normalize_url("http://example.com:80/foo") == "http://example.com/foo"
    assert normalize_url("https://example.com:443/foo") == "https://example.com/foo"


def test_normalize_url_keeps_non_default_port() -> None:
    assert normalize_url("https://example.com:8443/foo") == "https://example.com:8443/foo"


def test_normalize_url_strips_fragment_keeps_query() -> None:
    result = normalize_url("https://example.com/foo?b=2&a=1#section")
    assert result == "https://example.com/foo?b=2&a=1"
    assert "#" not in result


def test_normalize_url_empty_path_becomes_slash() -> None:
    assert normalize_url("https://example.com") == "https://example.com/"
    assert normalize_url("https://EXAMPLE.com") == "https://example.com/"


def test_cache_key_is_sha256_hex_of_normalized_url() -> None:
    import hashlib

    key = cache_key("HTTP://Example.com:80/Foo")
    expected = hashlib.sha256(
        normalize_url("HTTP://Example.com:80/Foo").encode("utf-8")
    ).hexdigest()
    assert key == expected
    assert len(key) == 64


def test_cache_key_same_for_equivalent_urls() -> None:
    assert cache_key("https://example.com") == cache_key("https://EXAMPLE.com:443/")


# --------------------------------------------------------------------------
# FetchResult.read_bytes
# --------------------------------------------------------------------------


def test_read_bytes_raises_when_not_ok() -> None:
    result = FetchResult(
        url="https://x.example/",
        final_url="https://x.example/",
        status=FetchStatus.TIMEOUT,
    )
    with pytest.raises(ValueError, match="status"):
        result.read_bytes()


def test_read_bytes_reads_content_path(tmp_path: Path) -> None:
    body = tmp_path / "body.bin"
    body.write_bytes(b"hello world")
    result = FetchResult(
        url="https://x.example/",
        final_url="https://x.example/",
        status=FetchStatus.OK,
        content_path=body,
    )
    assert result.read_bytes() == b"hello world"


# --------------------------------------------------------------------------
# Fetcher: cache
# --------------------------------------------------------------------------


async def test_cache_hit_skips_network(cfg: QuarryConfig) -> None:
    with respx.mock:
        _allow_robots("cache-test.example")
        page_route = respx.get("https://cache-test.example/page").mock(
            return_value=httpx.Response(200, html="<html><body>hi</body></html>")
        )
        fetcher = Fetcher(cfg.fetch)
        try:
            first = await fetcher.fetch("https://cache-test.example/page")
            second = await fetcher.fetch("https://cache-test.example/page")
        finally:
            await fetcher.aclose()

        assert first.status == FetchStatus.OK
        assert first.from_cache is False
        assert second.status == FetchStatus.OK
        assert second.from_cache is True
        assert page_route.call_count == 1
        assert first.read_bytes() == second.read_bytes()

        key = cache_key("https://cache-test.example/page")
        body_path = cfg.fetch.cache_dir / key[:2] / f"{key}.body"
        meta_path = cfg.fetch.cache_dir / key[:2] / f"{key}.meta.json"
        assert body_path.is_file()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["url"] == "https://cache-test.example/page"


# --------------------------------------------------------------------------
# Fetcher: robots
# --------------------------------------------------------------------------


async def test_robots_disallowed_blocks_fetch(cfg: QuarryConfig, fixtures_dir: Path) -> None:
    robots_text = _robots_text(fixtures_dir, "gridwatch-daily.example")
    with respx.mock:
        respx.get("https://gridwatch-daily.example/robots.txt").mock(
            return_value=httpx.Response(
                200, text=robots_text, headers={"content-type": "text/plain"}
            )
        )
        page_route = respx.get(
            "https://gridwatch-daily.example/private/f40-internal-briefing"
        ).mock(return_value=httpx.Response(200, html="<html>secret</html>"))
        fetcher = Fetcher(cfg.fetch)
        try:
            result = await fetcher.fetch(
                "https://gridwatch-daily.example/private/f40-internal-briefing"
            )
        finally:
            await fetcher.aclose()

        assert result.status == FetchStatus.ROBOTS_DISALLOWED
        assert result.content_path is None
        assert page_route.call_count == 0


async def test_robots_error_allows_fetch(cfg: QuarryConfig) -> None:
    with respx.mock:
        respx.get("https://robots-down.example/robots.txt").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        page_route = respx.get("https://robots-down.example/page").mock(
            return_value=httpx.Response(200, html="<html>ok</html>")
        )
        fetcher = Fetcher(cfg.fetch)
        try:
            result = await fetcher.fetch("https://robots-down.example/page")
        finally:
            await fetcher.aclose()

        assert result.status == FetchStatus.OK
        assert page_route.call_count == 1


# --------------------------------------------------------------------------
# Fetcher: guards
# --------------------------------------------------------------------------


async def test_too_large_response_rejected(cfg: QuarryConfig) -> None:
    cfg.fetch.max_bytes = 100
    with respx.mock:
        _allow_robots("huge-file.example")
        respx.get("https://huge-file.example/big").mock(
            return_value=httpx.Response(200, html="<html>" + ("x" * 1000) + "</html>")
        )
        fetcher = Fetcher(cfg.fetch)
        try:
            result = await fetcher.fetch("https://huge-file.example/big")
        finally:
            await fetcher.aclose()

        assert result.status == FetchStatus.TOO_LARGE
        assert result.content_path is None
        assert not list(cfg.fetch.cache_dir.rglob("*.body"))


async def test_non_html_content_type_rejected(cfg: QuarryConfig) -> None:
    with respx.mock:
        _allow_robots("api-host.example")
        respx.get("https://api-host.example/data").mock(
            return_value=httpx.Response(
                200, content=b'{"a": 1}', headers={"content-type": "application/json"}
            )
        )
        fetcher = Fetcher(cfg.fetch)
        try:
            result = await fetcher.fetch("https://api-host.example/data")
        finally:
            await fetcher.aclose()

        assert result.status == FetchStatus.NON_HTML
        assert result.content_type == "application/json"


async def test_timeout_maps_to_timeout_status(cfg: QuarryConfig) -> None:
    with respx.mock:
        _allow_robots("slow-host.example")
        respx.get("https://slow-host.example/slow").mock(
            side_effect=httpx.TimeoutException("timed out")
        )
        fetcher = Fetcher(cfg.fetch)
        try:
            result = await fetcher.fetch("https://slow-host.example/slow")
        finally:
            await fetcher.aclose()

        assert result.status == FetchStatus.TIMEOUT
        assert result.error


async def test_http_500_maps_to_http_error(cfg: QuarryConfig) -> None:
    with respx.mock:
        _allow_robots("broken-host.example")
        respx.get("https://broken-host.example/oops").mock(
            return_value=httpx.Response(500, html="<html>server error</html>")
        )
        fetcher = Fetcher(cfg.fetch)
        try:
            result = await fetcher.fetch("https://broken-host.example/oops")
        finally:
            await fetcher.aclose()

        assert result.status == FetchStatus.HTTP_ERROR
        assert result.http_status == 500
        assert not list(cfg.fetch.cache_dir.rglob("*.body"))


# --------------------------------------------------------------------------
# Fetcher: fetch_many, politeness headers
# --------------------------------------------------------------------------


async def test_fetch_many_preserves_input_order(cfg: QuarryConfig) -> None:
    urls = [
        "https://domain-a.example/page",
        "https://domain-b.example/page",
        "https://domain-c.example/page",
    ]
    with respx.mock:
        for i, host in enumerate(("domain-a.example", "domain-b.example", "domain-c.example")):
            _allow_robots(host)
            respx.get(f"https://{host}/page").mock(
                return_value=httpx.Response(200, html=f"<html>{i}</html>")
            )
        fetcher = Fetcher(cfg.fetch)
        try:
            results = await fetcher.fetch_many(urls)
        finally:
            await fetcher.aclose()

        assert [r.url for r in results] == urls
        assert all(r.status == FetchStatus.OK for r in results)


async def test_user_agent_header_is_sent(cfg: QuarryConfig) -> None:
    with respx.mock:
        _allow_robots("ua-check.example")
        page_route = respx.get("https://ua-check.example/page").mock(
            return_value=httpx.Response(200, html="<html>ok</html>")
        )
        fetcher = Fetcher(cfg.fetch)
        try:
            result = await fetcher.fetch("https://ua-check.example/page")
        finally:
            await fetcher.aclose()

        assert result.status == FetchStatus.OK
        sent_request = page_route.calls.last.request
        assert sent_request.headers["user-agent"] == cfg.fetch.user_agent
