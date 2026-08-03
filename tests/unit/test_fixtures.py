"""Corpus invariants: shape, synthetic provenance, determinism."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from make_fixtures import MARKER, build_corpus


def _manifest(fixtures_dir: Path) -> dict:
    return json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))


def test_corpus_shape(fixtures_dir: Path) -> None:
    manifest = _manifest(fixtures_dir)
    docs = manifest["docs"]
    assert len(docs) == 40
    kinds = {d["kind"] for d in docs}
    assert kinds == {
        "normal",
        "near_dup",
        "paywall",
        "js_only",
        "bad_encoding",
        "robots_disallowed",
    }
    assert sum(1 for d in docs if d["kind"] == "near_dup") == 6
    for doc in docs:
        assert (fixtures_dir / "html" / doc["file"]).is_file()


def test_all_urls_are_reserved_example_domains(fixtures_dir: Path) -> None:
    for doc in _manifest(fixtures_dir)["docs"]:
        assert doc["url"].startswith("https://")
        host = doc["url"].split("/")[2]
        assert host.endswith(".example"), f"{doc['file']} uses non-reserved domain {host}"


def test_every_html_file_carries_synthetic_marker(fixtures_dir: Path) -> None:
    for path in (fixtures_dir / "html").glob("*.html"):
        head = path.read_bytes()[:400].decode("utf-8", errors="replace")
        assert "SYNTHETIC-FIXTURE-CORPUS" in head, f"{path.name} lacks provenance marker"


def test_provenance_and_marker_files(fixtures_dir: Path) -> None:
    provenance = (fixtures_dir / "PROVENANCE.md").read_text(encoding="utf-8")
    assert MARKER in provenance
    assert MARKER in _manifest(fixtures_dir)["marker"]


def test_near_dups_share_most_content(fixtures_dir: Path) -> None:
    manifest = _manifest(fixtures_dir)
    pairs = [(d["file"], d["dup_of"]) for d in manifest["docs"] if d["kind"] == "near_dup"]
    assert pairs
    for dup_file, base_file in pairs:
        dup = (fixtures_dir / "html" / dup_file).read_text(encoding="utf-8")
        base = (fixtures_dir / "html" / base_file).read_text(encoding="utf-8")
        dup_words = dup.split()
        base_words = set(base.split())
        shared = sum(1 for w in dup_words if w in base_words)
        assert shared / len(dup_words) > 0.85, f"{dup_file} diverged too far from {base_file}"


def test_robots_fixture_disallows_private(fixtures_dir: Path) -> None:
    robots = (fixtures_dir / "robots" / "gridwatch-daily.example.txt").read_text(encoding="utf-8")
    assert "Disallow: /private/" in robots
    disallowed = [d for d in _manifest(fixtures_dir)["docs"] if d["kind"] == "robots_disallowed"]
    assert len(disallowed) == 1
    assert "/private/" in disallowed[0]["url"]


def test_bad_encoding_fixture_is_actually_malformed(fixtures_dir: Path) -> None:
    raw = (fixtures_dir / "html" / "f39.html").read_bytes()
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")


def test_relevance_set_holds_30_graded_chunks(fixtures_dir: Path) -> None:
    relevance = json.loads((fixtures_dir / "relevance.json").read_text(encoding="utf-8"))
    total = sum(len(q["chunks"]) for q in relevance["queries"])
    assert total == 30
    grades = {c["grade"] for q in relevance["queries"] for c in q["chunks"]}
    assert grades == {0, 1, 2, 3}


def test_anthropic_shapes_have_usage_and_no_identifiers(fixtures_dir: Path) -> None:
    for path in (fixtures_dir / "anthropic").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(payload)
        assert "sk-ant" not in text
        assert "org_" not in text
        if payload.get("type") == "message":
            usage = payload["usage"]
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ):
                assert key in usage
            assert payload["id"].startswith("msg_fixture_")
    read = json.loads(
        (fixtures_dir / "anthropic" / "synth_cache_read.json").read_text(encoding="utf-8")
    )
    assert read["usage"]["cache_read_input_tokens"] == 60000


def test_generation_is_deterministic(fixtures_dir: Path, tmp_path: Path) -> None:
    """Regenerate into a temp dir and hash-compare every file against the
    committed corpus. Drift between generator and corpus is a failure."""
    build_corpus(tmp_path)
    committed = sorted(p for p in fixtures_dir.rglob("*") if p.is_file())
    regenerated = sorted(p for p in tmp_path.rglob("*") if p.is_file())
    rel_committed = [p.relative_to(fixtures_dir) for p in committed]
    rel_regenerated = [p.relative_to(tmp_path) for p in regenerated]
    assert rel_committed == rel_regenerated
    for rel in rel_committed:
        a = hashlib.sha256((fixtures_dir / rel).read_bytes()).hexdigest()
        b = hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest()
        assert a == b, f"fixture drift in {rel}: regenerate with `make fixtures`"
