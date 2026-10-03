"""Fact-check domain blocklist (P0.7, rule 5: no leakage).

Search results from these domains are dropped before anything is graded: they
publish the verdicts our benchmark claims were drawn from, so letting them
through would leak the label rather than test retrieval.
"""

from __future__ import annotations

from urllib.parse import urlparse

BLOCKED_DOMAINS: frozenset[str] = frozenset(
    {
        # claim sources for AVeriTeC / FEVER-style benchmarks
        "politifact.com",
        "snopes.com",
        "factcheck.org",
        "fullfact.org",
        "africacheck.org",
        "checkyourfact.com",
        "leadstories.com",
        "truthorfiction.com",
        "boomlive.in",
        "altnews.in",
        "factly.in",
        "vishvasnews.com",
        "newsmeter.in",
        "thequint.com",  # WebQoof
        "logicallyfacts.com",
        "factcheck.afp.com",
        "afp.com",
        "reuters.com",  # Reuters Fact Check lives on the main domain
        "apnews.com",  # AP Fact Check, same
        "usatoday.com",  # USA Today Fact Check, same
        "washingtonpost.com",  # Fact Checker column
        "factcheckni.org",
        "verafiles.org",
        "rappler.com",
        "demagog.org.pl",
        "correctiv.org",
        "mimikama.org",
        "maldita.es",
        "newtral.es",
        "pagellapolitica.it",
        "teyit.org",
        "stopfake.org",
        "emergent.info",
        "climatefeedback.org",
        "healthfeedback.org",
        "sciencefeedback.co",
        "metafact.io",
        "factcheckhub.com",
        "dubawa.org",
        # the benchmarks themselves
        "fever.ai",
        "huggingface.co",
    }
)

# Blocked wherever they appear in the host, including inside a path-like subdomain.
BLOCKED_SUBSTRINGS: tuple[str, ...] = ("factcheck", "fact-check", "factchecking")


def domain_of(url: str) -> str:
    """Registrable-ish host: lowercased, no port, no leading `www.`."""
    host = urlparse(url if "//" in url else f"//{url}").hostname or ""
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def is_blocked(url: str) -> bool:
    """Fail closed: anything we cannot resolve to a plausible host is dropped."""
    host = domain_of(url)
    if not host or "." not in host or " " in host:
        return True
    if host in BLOCKED_DOMAINS:
        return True
    if any(host.endswith("." + d) for d in BLOCKED_DOMAINS):
        return True
    return any(s in host for s in BLOCKED_SUBSTRINGS)


def filter_urls(urls: list[str]) -> list[str]:
    return [u for u in urls if not is_blocked(u)]


def filter_results(results: list[dict], url_key: str = "url") -> list[dict]:
    """Drop search hits whose URL is blocked. Used by every search path."""
    return [r for r in results if not is_blocked(r.get(url_key, ""))]
