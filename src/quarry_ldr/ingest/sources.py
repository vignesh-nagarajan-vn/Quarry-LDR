"""Deterministic source tiers: how much a URL's publisher can be trusted by default.

The rating is code, not a model call, so it is free, reproducible, and safe on
resume. Unknown domains are ``general`` and are never punished; ``low`` is kept
for publisher types whose pages routinely repeat each other or are unvetted:
market-report mills, machine-generated vendor reports, and social media.
``source.overrides`` in config takes precedence over every built-in rule.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Mapping
from urllib.parse import urlsplit


class SourceTier(enum.StrEnum):
    PRIMARY = "primary"  # governments, regulators, intergovernmental and standards bodies
    SCHOLARLY = "scholarly"  # journals, preprint servers, universities
    ESTABLISHED = "established"  # reference works, major news, specialist trade press
    GENERAL = "general"  # everything unrecognized
    LOW = "low"  # report mills, machine-generated vendor reports, social media


STRONG_TIERS = frozenset({SourceTier.PRIMARY, SourceTier.SCHOLARLY})

_GOVERNMENT = re.compile(
    r"(^|\.)(gov|mil|int)$|\.(gov|mil)\.[a-z]{2}$|(^|\.)(europa\.eu|gc\.ca|gouv\.fr|go\.jp|govt\.nz)$"
)
_ACADEMIC = re.compile(r"(^|\.)edu$|\.(edu|ac)\.[a-z]{2}$")
_MARKET_REPORT = re.compile(
    r"market(s|data|research|report|reports|study|watch)|researchandmarkets|"
    r"mordorintelligence|grandviewresearch|fortunebusinessinsights|datamintelligence"
)
_PRIMARY_SITES = (
    "iea.org",
    "irena.org",
    "oecd.org",
    "worldbank.org",
    "imf.org",
    "un.org",
    "iso.org",
    "iec.ch",
    "ipcc.ch",
    "unep.org",
)
_SCHOLARLY_SITES = (
    "acm.org",
    "acs.org",
    "arxiv.org",
    "biorxiv.org",
    "bmj.com",
    "cambridge.org",
    "cell.com",
    "doi.org",
    "frontiersin.org",
    "ieee.org",
    "iop.org",
    "jamanetwork.com",
    "jstor.org",
    "mdpi.com",
    "medrxiv.org",
    "nature.com",
    "nejm.org",
    "oup.com",
    "plos.org",
    "repec.org",
    "researchgate.net",
    "rsc.org",
    "sagepub.com",
    "science.org",
    "sciencedirect.com",
    "semanticscholar.org",
    "springer.com",
    "ssrn.com",
    "tandfonline.com",
    "thelancet.com",
    "wiley.com",
)
_ESTABLISHED_SITES = (
    "apnews.com",
    "bbc.co.uk",
    "bbc.com",
    "bloomberg.com",
    "britannica.com",
    "canarymedia.com",
    "carbonbrief.org",
    "economist.com",
    "energy-storage.news",
    "ft.com",
    "latitudemedia.com",
    "npr.org",
    "nytimes.com",
    "pv-magazine.com",
    "reuters.com",
    "statnews.com",
    "technologyreview.com",
    "theguardian.com",
    "utilitydive.com",
    "washingtonpost.com",
    "wikipedia.org",
    "wsj.com",
)
_LOW_SITES = (
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "patsnap.com",
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "tiktok.com",
    "x.com",
    "youtube.com",
)
_TWO_PART_SUFFIX_LABELS = frozenset({"ac", "co", "com", "edu", "gov", "net", "or", "org"})


def domain_of(url: str) -> str:
    """Lowercase hostname without a leading ``www.``."""
    host = (urlsplit(url).hostname or "").lower()
    return host.removeprefix("www.")


def site_of(url: str) -> str:
    """The publisher a URL belongs to: ``eureka.patsnap.com`` and
    ``patsnap.com`` are one site for concentration limits."""
    labels = domain_of(url).split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in _TWO_PART_SUFFIX_LABELS:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _matches(host: str, sites: tuple[str, ...]) -> bool:
    return any(host == site or host.endswith(f".{site}") for site in sites)


def rate_url(url: str, overrides: Mapping[str, str] | None = None) -> SourceTier:
    """The tier for ``url``. An override key matches a host or any subdomain."""
    host = domain_of(url)
    if overrides:
        # Longest key first, so a subdomain override beats its parent's.
        for site in sorted(overrides, key=len, reverse=True):
            if host == site or host.endswith(f".{site}"):
                return SourceTier(overrides[site])
    if _matches(host, _LOW_SITES) or _MARKET_REPORT.search(host):
        return SourceTier.LOW
    if _GOVERNMENT.search(host) or _matches(host, _PRIMARY_SITES):
        return SourceTier.PRIMARY
    if _ACADEMIC.search(host) or _matches(host, _SCHOLARLY_SITES):
        return SourceTier.SCHOLARLY
    if _matches(host, _ESTABLISHED_SITES):
        return SourceTier.ESTABLISHED
    return SourceTier.GENERAL
