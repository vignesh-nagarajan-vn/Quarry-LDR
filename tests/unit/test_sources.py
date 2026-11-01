"""Source tiers are deterministic, overridable, and never punish the unknown."""

from __future__ import annotations

import pytest

from quarry_ldr.ingest.sources import SourceTier, domain_of, rate_url, site_of


@pytest.mark.parametrize(
    ("url", "tier"),
    [
        ("https://www.energy.gov/report", SourceTier.PRIMARY),
        ("https://assets.publishing.service.gov.uk/a.pdf", SourceTier.PRIMARY),
        ("https://www.who.int/news", SourceTier.PRIMARY),
        ("https://ec.europa.eu/energy", SourceTier.PRIMARY),
        ("https://www.iea.org/reports/x", SourceTier.PRIMARY),
        ("https://www.nature.com/articles/x", SourceTier.SCHOLARLY),
        ("https://advanced.onlinelibrary.wiley.com/doi/x", SourceTier.SCHOLARLY),
        ("https://web.mit.edu/x", SourceTier.SCHOLARLY),
        ("https://www.ox.ac.uk/x", SourceTier.SCHOLARLY),
        ("https://en.wikipedia.org/wiki/Battery", SourceTier.ESTABLISHED),
        ("https://www.reuters.com/x", SourceTier.ESTABLISHED),
        ("https://eureka.patsnap.com/report-x", SourceTier.LOW),
        ("https://www.polarismarketresearch.com/x", SourceTier.LOW),
        ("https://www.marketdataforecast.com/x", SourceTier.LOW),
        ("https://www.reddit.com/r/x", SourceTier.LOW),
        ("https://sunlithenergy.com/x", SourceTier.GENERAL),
        ("https://example.example/x", SourceTier.GENERAL),
    ],
)
def test_rate_url_tiers(url: str, tier: SourceTier) -> None:
    assert rate_url(url) is tier


def test_overrides_beat_built_in_rules_and_cover_subdomains() -> None:
    overrides = {"patsnap.com": "general", "eureka.patsnap.com": "scholarly"}
    assert rate_url("https://www.patsnap.com/x", overrides) is SourceTier.GENERAL
    assert rate_url("https://eureka.patsnap.com/x", overrides) is SourceTier.SCHOLARLY
    assert rate_url("https://mail.eureka.patsnap.com/x", overrides) is SourceTier.SCHOLARLY


def test_lookalike_hosts_do_not_inherit_a_tier() -> None:
    assert rate_url("https://notnature.com/x") is SourceTier.GENERAL
    assert rate_url("https://nature.com.example/x") is SourceTier.GENERAL


def test_site_groups_subdomains_of_one_publisher() -> None:
    assert site_of("https://eureka.patsnap.com/x") == site_of("https://www.patsnap.com/y")
    assert site_of("https://www.bbc.co.uk/x") == "bbc.co.uk"
    assert site_of("https://a.b.example.com/x") == "example.com"
    assert domain_of("https://WWW.Example.com/x") == "example.com"
