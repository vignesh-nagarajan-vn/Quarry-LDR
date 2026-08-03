"""Unit tests for near-duplicate removal: SimHash shingling and DedupResult.

The corpus-level regression at the bottom exercises the real extract/chunk
pipeline (read-only) over the near_dup fixtures and checks the SimHash-only
path (``embeddings=None``) actually earns its keep on realistic text.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from quarry_ldr.config import ChunkSettings, DedupSettings
from quarry_ldr.ingest.chunk import Chunk, chunk_document
from quarry_ldr.ingest.dedup import DedupResult, dedup_chunks, hamming, simhash64
from quarry_ldr.ingest.extract import extract_document


def _chunk(position: int, text: str, url: str = "https://a.example") -> Chunk:
    """Minimal Chunk for unit tests that don't need the real chunker."""
    return Chunk(
        chunk_id=f"c{position}",
        url=url,
        text=text,
        token_count=max(1, len(text) // 4),
        position=position,
        start_char=0,
        end_char=len(text),
    )


# A realistic-length paragraph (162 words) used to probe SimHash's tolerance
# to a light, single-word synonym swap versus a wholly unrelated paragraph.
_ENERGY_PARAGRAPH = (
    "The grid scale sand heat battery project near Vesterholm has secured financing from "
    "several regional partners this quarter, and construction is expected to begin soon at "
    "the coastal site. Local officials say the plan has strong community backing throughout "
    "the surrounding region, built up over several years of public meetings and environmental "
    "review. Engineers are finalizing the thermal storage design while utility officials "
    "review the interconnection plan ahead of the coming winter season. The developer expects "
    "the facility to store enough heat to supply the district network for several days during "
    "periods of peak demand, reducing reliance on imported natural gas for heating during the "
    "coldest months of the year. Analysts following the project say the financing structure "
    "could serve as a template for similar thermal storage projects elsewhere on the continent, "
    "particularly in regions with abundant industrial waste heat and strong district heating "
    "networks already in place, though permitting timelines remain a persistent source of "
    "uncertainty for developers weighing new sites."
)

_AIRLINE_PARAGRAPH = (
    "Quarterly earnings for the regional airline exceeded analyst expectations as passenger "
    "volume rebounded strongly across every major domestic route this season. Management "
    "pointed to resilient leisure travel demand and lower fuel costs as the primary drivers of "
    "the improved margins, and raised full year guidance for the third consecutive quarter. The "
    "airline also announced plans to expand its fleet with several new narrow body aircraft over "
    "the next two years, aiming to add capacity on its busiest transcontinental routes while "
    "retiring older, less efficient planes from regional service. Industry observers noted that "
    "the results stood in contrast to weaker performance from several rival carriers, which have "
    "struggled with higher labor costs and softer demand on international long haul routes "
    "serving business travelers during the slower summer months."
)


# --------------------------------------------------------------------------
# simhash64
# --------------------------------------------------------------------------


def test_simhash_identical_text_is_equal() -> None:
    assert simhash64(_ENERGY_PARAGRAPH) == simhash64(_ENERGY_PARAGRAPH)
    assert hamming(simhash64(_ENERGY_PARAGRAPH), simhash64(_ENERGY_PARAGRAPH)) == 0


def test_simhash_is_case_and_whitespace_insensitive() -> None:
    a = simhash64("Hello World, this is Test Text")
    b = simhash64("hello   world,   this is   test text")
    assert a == b
    assert a != 0


def test_simhash_short_text_hashes_whole_word_list_as_one_shingle() -> None:
    # Both texts are shorter than the default shingle_size=5, so each hashes
    # its entire word list as a single shingle; different word lists should
    # (with overwhelming probability) hash differently.
    short_a = simhash64("alpha beta")
    short_b = simhash64("alpha beta gamma")
    assert short_a != short_b
    assert short_a != 0
    assert short_b != 0


def test_simhash_empty_and_whitespace_text_returns_zero() -> None:
    assert simhash64("") == 0
    assert simhash64("   \n\t  ") == 0


def test_simhash_light_synonym_swap_stays_close_unrelated_text_is_far() -> None:
    swapped = _ENERGY_PARAGRAPH.replace("expected", "planned", 1)
    assert swapped != _ENERGY_PARAGRAPH  # sanity: the swap actually changed the text

    swap_distance = hamming(simhash64(_ENERGY_PARAGRAPH), simhash64(swapped))
    unrelated_distance = hamming(simhash64(_ENERGY_PARAGRAPH), simhash64(_AIRLINE_PARAGRAPH))

    assert swap_distance <= 3
    assert unrelated_distance > 10


# --------------------------------------------------------------------------
# hamming
# --------------------------------------------------------------------------


def test_hamming_basic_properties() -> None:
    assert hamming(0, 0) == 0
    assert hamming(0b1010, 0b1010) == 0
    assert hamming(0b0000, 0b0001) == 1
    assert hamming(0b0000, 0b1111) == 4


def test_hamming_symmetric_and_full_width() -> None:
    a, b = simhash64(_ENERGY_PARAGRAPH), simhash64(_AIRLINE_PARAGRAPH)
    assert hamming(a, b) == hamming(b, a)
    all_ones = (1 << 64) - 1
    assert hamming(0, all_ones) == 64


# --------------------------------------------------------------------------
# dedup_chunks: ordering, matching semantics
# --------------------------------------------------------------------------


def test_dedup_first_wins_keeps_earlier_of_duplicate_pair() -> None:
    chunks = [_chunk(0, _ENERGY_PARAGRAPH), _chunk(1, _ENERGY_PARAGRAPH)]
    result = dedup_chunks(chunks, None, DedupSettings())
    assert result.kept == [0]
    assert result.dropped == {1: 0}


def test_dedup_first_wins_matches_the_correct_earlier_chunk() -> None:
    # index 2 duplicates index 0, not the unrelated chunk sitting between them.
    chunks = [
        _chunk(0, _ENERGY_PARAGRAPH),
        _chunk(1, _AIRLINE_PARAGRAPH),
        _chunk(2, _ENERGY_PARAGRAPH),
    ]
    result = dedup_chunks(chunks, None, DedupSettings())
    assert result.kept == [0, 1]
    assert result.dropped == {2: 0}


def test_dedup_zero_simhash_hashes_are_not_falsely_matched() -> None:
    # Both chunks hash to 0 (empty text); hamming(0, 0) == 0 <= any threshold,
    # but the "both hashes nonzero" guard must stop this being a false match.
    chunks = [_chunk(0, ""), _chunk(1, "   ")]
    result = dedup_chunks(chunks, None, DedupSettings())
    assert result.kept == [0, 1]
    assert result.dropped == {}


def test_dedup_cosine_path_catches_paraphrase_simhash_misses() -> None:
    # Two nearly-parallel normalized embedding rows (dot >= 0.92) paired with
    # completely different wording, so SimHash alone would not flag them.
    theta_cos, theta_sin = 0.95, math.sqrt(1 - 0.95**2)
    embeddings = np.array(
        [[1.0, 0.0, 0.0, 0.0], [theta_cos, theta_sin, 0.0, 0.0]], dtype=np.float32
    )
    assert float(np.dot(embeddings[0], embeddings[1])) >= 0.92

    chunks = [_chunk(0, _ENERGY_PARAGRAPH), _chunk(1, _AIRLINE_PARAGRAPH)]
    settings = DedupSettings()
    # Confirm the premise: SimHash alone would have missed this pair.
    assert hamming(simhash64(_ENERGY_PARAGRAPH), simhash64(_AIRLINE_PARAGRAPH)) > (
        settings.simhash_hamming_max
    )

    result = dedup_chunks(chunks, embeddings, settings)
    assert result.kept == [0]
    assert result.dropped == {1: 0}


def test_dedup_embeddings_none_runs_simhash_only() -> None:
    swapped = _ENERGY_PARAGRAPH.replace("expected", "planned", 1)
    chunks = [
        _chunk(0, _ENERGY_PARAGRAPH),
        _chunk(1, swapped),  # near-dup by SimHash
        _chunk(2, _AIRLINE_PARAGRAPH),  # unrelated
    ]
    result = dedup_chunks(chunks, None, DedupSettings())
    assert result.kept == [0, 2]
    assert result.dropped == {1: 0}


def test_dedup_drop_rate_math() -> None:
    chunks = [
        _chunk(0, _ENERGY_PARAGRAPH),
        _chunk(1, _ENERGY_PARAGRAPH),
        _chunk(2, _AIRLINE_PARAGRAPH),
        _chunk(3, _ENERGY_PARAGRAPH),
        _chunk(4, _AIRLINE_PARAGRAPH),
    ]
    result = dedup_chunks(chunks, None, DedupSettings())
    assert result.n_input == 5
    assert result.n_kept == 2
    assert result.drop_rate == 1.0 - (2 / 5)


def test_dedup_result_drop_rate_property_direct() -> None:
    result = DedupResult(kept=[0, 2, 4], dropped={1: 0, 3: 2}, n_input=5, n_kept=3)
    assert result.drop_rate == 1.0 - (3 / 5)


def test_dedup_empty_input() -> None:
    result = dedup_chunks([], None, DedupSettings())
    assert result.kept == []
    assert result.dropped == {}
    assert result.n_input == 0
    assert result.n_kept == 0
    assert result.drop_rate == 0.0


# --------------------------------------------------------------------------
# Corpus-level regression: real extract -> chunk -> dedup over the fixtures.
# --------------------------------------------------------------------------

# SimHash-only near-duplicate detection at chunk granularity (~100-200 words
# per chunk, shingle_size=5) is noisier than at document granularity: a
# handful of scattered synonym swaps across a chunk can nudge several bits
# past a tight radius purely from SimHash's bit-vote variance. The shipped
# default (simhash_hamming_max=3) is tuned for the SimHash+cosine combo, not
# SimHash alone; a wider radius is used here to exercise the SimHash-only
# path meaningfully. It is still checked against the fixture corpus's actual
# separation: no two chunks from genuinely unrelated normal docs land under
# 13 bits apart, so 10 leaves a comfortable margin on both sides.
_CORPUS_SIMHASH_ONLY_SETTINGS = DedupSettings(simhash_hamming_max=10)


def _load_manifest(fixtures_dir: Path) -> list[dict[str, object]]:
    data = json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))
    return list(data["docs"])


def _entry(fixtures_dir: Path, filename: str) -> dict[str, object]:
    for doc in _load_manifest(fixtures_dir):
        if doc["file"] == filename:
            return doc
    raise AssertionError(f"{filename} not found in manifest.json")


def _extract_and_chunk(fixtures_dir: Path, filename: str) -> list[Chunk]:
    entry = _entry(fixtures_dir, filename)
    html = (fixtures_dir / "html" / filename).read_bytes()
    doc = extract_document(html, str(entry["url"]))
    assert doc is not None, f"{filename} failed to extract"
    return chunk_document(doc, ChunkSettings())


def test_corpus_near_dup_chunks_mostly_dropped_via_simhash(fixtures_dir: Path) -> None:
    manifest = _load_manifest(fixtures_dir)
    near_dup_pairs = [
        (str(doc["dup_of"]), str(doc["file"])) for doc in manifest if doc["kind"] == "near_dup"
    ]
    assert len(near_dup_pairs) == 6  # matches the fixture corpus's fixed shape

    for base_file, dup_file in near_dup_pairs:
        base_chunks = _extract_and_chunk(fixtures_dir, base_file)
        dup_chunks = _extract_and_chunk(fixtures_dir, dup_file)
        assert base_chunks and dup_chunks

        combined = base_chunks + dup_chunks
        result = dedup_chunks(combined, None, _CORPUS_SIMHASH_ONLY_SETTINGS)

        n_base = len(base_chunks)
        dup_indices = range(n_base, n_base + len(dup_chunks))
        dropped_as_dup_of_base = [
            i for i in dup_indices if i in result.dropped and result.dropped[i] < n_base
        ]
        drop_rate = len(dropped_as_dup_of_base) / len(dup_chunks)
        assert drop_rate >= 0.6, (
            f"{dup_file} only dropped {drop_rate:.0%} of its chunks as duplicates of {base_file}"
        )


def test_corpus_unrelated_normal_docs_not_flagged_as_duplicates(fixtures_dir: Path) -> None:
    # f01 and f04 are both kind="normal" with dup_of=None and cover distinct
    # topics -- nothing in either should be flagged as a duplicate of the other.
    first_chunks = _extract_and_chunk(fixtures_dir, "f01.html")
    second_chunks = _extract_and_chunk(fixtures_dir, "f04.html")
    assert first_chunks and second_chunks

    result = dedup_chunks(first_chunks + second_chunks, None, _CORPUS_SIMHASH_ONLY_SETTINGS)
    assert result.dropped == {}
    assert result.n_kept == result.n_input
