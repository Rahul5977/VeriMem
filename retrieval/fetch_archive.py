"""Build a knowledge store from AVeriTeC's archived source pages (P0.8 fallback).

The official prebuilt knowledge store was withdrawn, and the original repo only ships a
script that rebuilds one through the Google Search API. But the gold annotations already
point almost entirely at Wayback Machine snapshots -- 961 unique `web.archive.org` URLs
across the 500 dev claims -- and those are permanent, free, and reproducible by anyone
with the dataset. So we fetch them and index the lot.

Two things this is **not**:

* It is not the official store. That one also held distractor pages from Google search,
  so retrieval there is harder. Absolute numbers from this store are not comparable to
  published AVeriTeC results. Internal comparisons are unaffected -- every system we
  report uses the same pool.
* It is not per-claim. Documents go into one shared pool, so a claim's two or three gold
  pages sit among hundreds of others and retrieval has to actually work. Building a pool
  from only the claims under evaluation would make the baseline look far better than it
  is.

Every fetch goes through the hashed cache (rule 1), so the job is resumable and a second
run costs nothing.

    python -m retrieval.fetch_archive --data data/averitec/dev.json --limit 20
    python -m retrieval.fetch_archive --data data/averitec/dev.json   # the full pool
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.cache import get_cache
from core.paths import DATA_DIR
from eval.datasets import Claim, load_averitec, sample
from retrieval.blocklist import is_blocked

USER_AGENT = (
    "Mozilla/5.0 (compatible; VeriMem academic research; +https://github.com/TERABYTE05/VeriMem)"
)
MAX_BYTES = 2_000_000
TIMEOUT = 60

# Tags whose text is never content.
SKIP_TAGS = frozenset(
    {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "button"}
)
# The Wayback Machine injects its own toolbar into every snapshot. Left in, it becomes
# the most common text in the corpus and pollutes every BM25 score.
WAYBACK_IDS = frozenset({"wm-ipp-base", "wm-ipp", "wm-ipp-print", "donato"})

_WS = re.compile(r"\s+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


class _TextExtractor(HTMLParser):
    """Minimal readable-text extraction. No dependencies beyond the standard library."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._skip_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        skip = tag in SKIP_TAGS or (attributes.get("id") or "") in WAYBACK_IDS
        if skip and self._skip_depth == 0:
            self._skip_tag, self._skip_depth = tag, 1
        elif self._skip_depth and tag == self._skip_tag:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth and tag == self._skip_tag:
            self._skip_depth -= 1
            if self._skip_depth == 0:
                self._skip_tag = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = _WS.sub(" ", data).strip()
        if len(text) > 1:
            self.parts.append(text)


def extract_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:  # malformed markup; keep whatever was parsed
        pass
    return _WS.sub(" ", " ".join(parser.parts)).strip()


def split_sentences(text: str, min_len: int = 30, max_sentences: int = 400) -> list[str]:
    """Sentence-ish split. Short fragments are navigation chrome, not evidence."""
    out = []
    for piece in _SENTENCE.split(text):
        piece = piece.strip()
        if len(piece) >= min_len:
            out.append(piece[:1500])
        if len(out) >= max_sentences:
            break
    return out


def archived_urls(claims: list[Claim], archived_only: bool = True) -> list[str]:
    """Unique evidence URLs across the claims, preferring the archived snapshot."""
    urls: dict[str, None] = {}
    for claim in claims:
        for question in claim.questions:
            for answer in question.get("answers", []) or []:
                url = answer.get("cached_source_url") or answer.get("source_url") or ""
                if not url.startswith("http") or is_blocked(url):
                    continue
                if archived_only and "web.archive.org" not in url:
                    continue
                urls.setdefault(url, None)
    return list(urls)


def fetch_url(url: str, polite_delay: float = 1.0, attempts: int = 3) -> str | None:
    """Fetch one page. Successes are cached; failures deliberately are not.

    `get_or_compute` would happily cache a None, which turns one transient timeout or
    rate-limit into a permanent hole in the pool that no re-run ever fills. Over
    thousands of URLs against a service that throttles, that is the difference between a
    resumable job and a quietly incomplete corpus. So a failure returns None without
    being written, and the next run tries again.

    HTTP 404 and 403 are treated as permanent and not retried; everything else gets a
    backoff, since the archive rate-limits under load.
    """
    cache = get_cache("fetch")
    key = {"url": url}
    hit = cache.get(key)
    if hit is not None:
        return hit

    for attempt in range(attempts):
        time.sleep(polite_delay * (0.5 + random.random()) + attempt * 4)
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=TIMEOUT) as response:
                raw = response.read(MAX_BYTES)
            charset = response.headers.get_content_charset() or "utf-8"
            text = raw.decode(charset, errors="replace")
            cache.set(key, text)
            return text
        except HTTPError as exc:
            if exc.code in (403, 404, 410):
                return None  # the snapshot is genuinely gone; retrying cannot help
        except (URLError, TimeoutError, OSError, ValueError):
            pass
    return None


def build_pool(
    urls: list[str],
    out_path: Path,
    workers: int = 4,
    polite_delay: float = 1.0,
) -> dict[str, int]:
    """Fetch every URL and write one JSONL row per page: {url, url2text}."""
    stats = {"requested": len(urls), "fetched": 0, "failed": 0, "empty": 0, "sentences": 0}
    rows: list[dict[str, object]] = []

    def work(url: str) -> tuple[str, list[str]]:
        html = fetch_url(url, polite_delay)
        if not html:
            return url, []
        return url, split_sentences(extract_text(html))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (url, sentences) in enumerate(pool.map(work, urls), start=1):
            if not sentences:
                stats["failed" if not fetch_url(url, 0) else "empty"] += 1
            else:
                stats["fetched"] += 1
                stats["sentences"] += len(sentences)
                rows.append({"url": url, "url2text": sentences})
            if i % 25 == 0 or i == len(urls):
                print(
                    f"  [{i}/{len(urls)}] ok={stats['fetched']} "
                    f"failed={stats['failed']} empty={stats['empty']}"
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m retrieval.fetch_archive")
    p.add_argument("--data", type=Path, required=True, help="AVeriTeC split")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="only this many claims' URLs")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--delay", type=float, default=1.0, help="polite delay per request")
    p.add_argument("--include-live", action="store_true", help="also try non-archived URLs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    claims = load_averitec(args.data)
    selected = sample(claims, args.limit, args.seed) if args.limit else claims
    urls = archived_urls(selected, archived_only=not args.include_live)

    out = args.out or DATA_DIR / "averitec" / f"pool_{args.data.stem}.jsonl"
    print(f"[verimem] {len(selected)} claims -> {len(urls)} unique archived URLs")
    est = len(urls) * (20 + args.delay) / args.workers / 60
    print(f"[verimem] ~{est:.0f} min at {args.workers} workers (cached fetches are instant)")
    if args.limit:
        print("[verimem] NOTE: a pool built from a subset makes retrieval artificially easy;")
        print("[verimem]       use the full split before reporting any baseline number.")
    if args.dry_run:
        return 0

    t0 = time.time()
    stats = build_pool(urls, out, args.workers, args.delay)
    print(f"\n[verimem] {stats} in {time.time() - t0:.0f}s")
    print(f"[verimem] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
