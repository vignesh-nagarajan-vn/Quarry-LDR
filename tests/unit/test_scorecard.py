"""The scorecard measures a report from markdown alone and gates on thresholds."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from quarry_ldr.cli import app
from quarry_ldr.report.scorecard import evaluate, load_thresholds, score_report

REPORT = """# Topic

## Key findings

Form Energy built a 1.5 MW plant in Minnesota [1]. Costs reached 40 percent lower levels [2].

## What are the main challenges in the field of

What are the main challenges today? Corrosion limits life across most of the tested designs [1].
This uncited sentence about the technology makes a long general claim with no source at all.
This uncited sentence about the technology makes a long general claim with no source at all.
Odd marker here [2 (sq02)] and another [1, 2].

## Sources

Tier mix goes here [1].

## References

[1] A paper - https://www.nature.com/articles/x (chunk a, chars 0-10)
[2] A report - https://eureka.patsnap.com/report-x (chunk b, chars 0-10)

## Cost ledger

**Total: $1.2500** (cap $5.00)

## Run manifest

```json
{"started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T00:30:00+00:00"}
```
"""


def test_required_section_names_are_not_cut_off_titles() -> None:
    card = score_report(
        "# T\n\n## What we could not establish\n\nA gap.\n\n## How the costs of\n\nA."
    )
    assert card.cut_off_titles == 1  # only the heading that really stops mid-phrase


def test_the_disagreement_section_is_not_a_cut_off_title() -> None:
    card = score_report("# T\n\n## Where sources disagree\n\n- **Cost:** A [1], B [2].")
    assert card.cut_off_titles == 0


def test_score_report_measures_the_report() -> None:
    card = score_report(REPORT)
    assert card.sections == 3
    assert card.citations == 3  # Sources is not prose; malformed markers do not count
    assert card.distinct_sites == 2
    assert card.top_site_share == pytest.approx(0.667)
    assert card.strong_source_share == pytest.approx(0.667)
    assert card.low_source_share == pytest.approx(0.333)
    assert card.malformed_markers == 2
    assert card.cut_off_titles == 1
    assert card.question_sentences == 1
    assert card.repeated_sentences == 1
    assert card.uncited_factual_share > 0
    assert card.missing_required_sections == 1
    assert card.cost_usd == 1.25
    assert card.minutes == 30.0
    assert card.specifics_per_1k_words > 0


def test_overrides_change_the_tier_mix() -> None:
    card = score_report(REPORT, {"patsnap.com": "scholarly"})
    assert card.low_source_share == 0.0
    assert card.strong_source_share == 1.0


def test_empty_report_scores_zero_without_error() -> None:
    card = score_report("# Nothing\n")
    assert card.words == 0
    assert card.top_site_share == 0.0
    assert card.missing_required_sections == 3


def test_evaluate_applies_min_max_and_warn_only() -> None:
    card = score_report(REPORT)
    results = {
        r.metric: r
        for r in evaluate(
            card,
            {
                "malformed_markers": {"max": 0},
                "citations": {"min": 1},
                "words": {"min": 10000, "warn_only": True},
            },
        )
    }
    assert not results["malformed_markers"].passed
    assert results["citations"].passed
    assert not results["words"].passed and results["words"].warn_only


def test_unknown_metric_raises() -> None:
    with pytest.raises(KeyError):
        evaluate(score_report(REPORT), {"not_a_metric": {"max": 1}})


def test_shipped_thresholds_name_real_metrics() -> None:
    evaluate(score_report(REPORT), load_thresholds())


def test_cli_score_reports_misses_and_compares(tmp_path: Path) -> None:
    path = tmp_path / "r.md"
    path.write_text(REPORT, encoding="utf-8")
    result = CliRunner().invoke(app, ["score", str(path), "--against", str(path)])
    assert result.exit_code == 1
    assert "MISS" in result.output
    assert "Before" in result.output


def test_cli_score_unknown_target(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["score", str(tmp_path / "nope")])
    assert result.exit_code == 2
