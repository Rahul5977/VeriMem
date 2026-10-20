"""Offline tests for the archive fetcher and the pooled store. No network."""

import json

from eval.datasets import Claim
from retrieval.fetch_archive import archived_urls, extract_text, split_sentences
from retrieval.store import PooledKnowledgeStore

WAYBACK_PAGE = """
<html><head><title>T</title>
<script>var tracking = "this must not appear";</script>
<style>.x { color: red; }</style></head>
<body>
<div id="wm-ipp-base"><div>INTERNET ARCHIVE Wayback Machine toolbar noise</div></div>
<nav>Home About Contact</nav>
<p>The committee published its final report in March 2019.</p>
<p>Officials confirmed the figure had been revised downward.</p>
<footer>Copyright notice</footer>
</body></html>
"""


def claim_with(urls, claim_id="c1"):
    return Claim(
        id=claim_id,
        claim="a claim",
        questions=[{"question": "q", "answers": [{"answer": "a", "source_url": u} for u in urls]}],
    )


# --- extraction ---------------------------------------------------------------------


def test_extracts_body_text():
    text = extract_text(WAYBACK_PAGE)
    assert "committee published its final report" in text
    assert "revised downward" in text


def test_script_and_style_are_dropped():
    text = extract_text(WAYBACK_PAGE)
    assert "this must not appear" not in text
    assert "color: red" not in text


def test_wayback_toolbar_is_dropped():
    """Left in, it is the most common text in the corpus and skews every BM25 score."""
    assert "Wayback Machine toolbar" not in extract_text(WAYBACK_PAGE)


def test_navigation_chrome_is_dropped():
    text = extract_text(WAYBACK_PAGE)
    assert "Home About Contact" not in text
    assert "Copyright notice" not in text


def test_malformed_html_does_not_raise():
    assert isinstance(extract_text("<p>unclosed <div><span>text"), str)


def test_empty_html():
    assert extract_text("") == ""


# --- sentence splitting -------------------------------------------------------------


def test_splits_on_sentence_boundaries():
    text = "The report was published in March. Officials confirmed the revised figure later."
    assert len(split_sentences(text, min_len=10)) == 2


def test_short_fragments_are_dropped():
    assert split_sentences("Home. Next. OK.", min_len=30) == []


def test_sentence_cap_is_respected():
    text = " ".join(
        f"This is sentence number {i} and it is long enough to keep." for i in range(50)
    )
    assert len(split_sentences(text, max_sentences=10)) == 10


# --- URL collection -----------------------------------------------------------------


def test_collects_and_dedupes_archived_urls():
    u = "https://web.archive.org/web/2020/https://example.org/a"
    urls = archived_urls([claim_with([u, u]), claim_with([u], "c2")])
    assert urls == [u]


def test_non_archived_urls_are_skipped_by_default():
    urls = archived_urls([claim_with(["https://example.org/live"])])
    assert urls == []


def test_non_archived_urls_included_on_request():
    urls = archived_urls([claim_with(["https://example.org/live"])], archived_only=False)
    assert urls == ["https://example.org/live"]


def test_blocked_domains_are_never_fetched():
    """Rule 5 applies before the request, not just at retrieval."""
    blocked = "https://web.archive.org/web/2020/https://www.politifact.com/x"
    assert archived_urls([claim_with([blocked])], archived_only=False) == []


def test_cached_source_url_is_preferred_over_source_url():
    claim = Claim(
        id="c1",
        claim="x",
        questions=[
            {
                "answers": [
                    {
                        "answer": "a",
                        "source_url": "https://web.archive.org/live",
                        "cached_source_url": "https://web.archive.org/cached",
                    }
                ]
            }
        ],
    )
    assert archived_urls([claim]) == ["https://web.archive.org/cached"]


# --- the pooled store ---------------------------------------------------------------


def pool_file(tmp_path, rows):
    path = tmp_path / "pool.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_pool_searches_every_document_regardless_of_claim_id(tmp_path):
    path = pool_file(
        tmp_path,
        [
            {"url": "https://a.org/1", "url2text": ["The Kochi metro opened in June 2017."] * 4},
            {"url": "https://b.org/1", "url2text": ["Penguins are flightless birds."] * 4},
        ],
    )
    store = PooledKnowledgeStore(path)
    # claim_id is ignored on purpose: one shared corpus, not a per-claim set
    for claim_id in ("1", "999", "anything"):
        hits = store.retrieve(claim_id, "when did the Kochi metro open", k=1)
        assert hits and hits[0].url == "https://a.org/1"


def test_pool_counts_documents_and_passages(tmp_path):
    path = pool_file(
        tmp_path,
        [
            {"url": f"https://a.org/{i}", "url2text": ["s one.", "s two.", "s three."]}
            for i in range(3)
        ],
    )
    store = PooledKnowledgeStore(path)
    assert store.n_documents == 3
    assert len(store) == 3  # 3 sentences each -> one passage per document


def test_pool_applies_the_blocklist(tmp_path):
    path = pool_file(
        tmp_path,
        [
            {"url": "https://www.politifact.com/x", "url2text": ["metro metro metro metro"] * 4},
            {"url": "https://ok.org/1", "url2text": ["metro opened in 2017."] * 4},
        ],
    )
    hits = PooledKnowledgeStore(path).retrieve("1", "metro", k=5)
    assert hits and all("politifact" not in h.url for h in hits)


def test_pool_tags_provenance(tmp_path):
    path = pool_file(tmp_path, [{"url": "https://a.org/1", "url2text": ["metro opened 2017."] * 4}])
    hits = PooledKnowledgeStore(path).retrieve("1", "metro", k=1)
    assert hits[0].retrieved_by == "archive_pool"


def test_empty_pool_is_safe(tmp_path):
    assert PooledKnowledgeStore(pool_file(tmp_path, [])).retrieve("1", "q", k=5) == []


# --- store auto-detection (a pooled file in KnowledgeStore retrieves nothing) -------


def test_open_store_detects_a_pooled_file(tmp_path):
    from retrieval.store import PooledKnowledgeStore, open_store

    path = pool_file(tmp_path, [{"url": "https://a.org/1", "url2text": ["metro opened."] * 4}])
    assert isinstance(open_store(path), PooledKnowledgeStore)


def test_open_store_detects_a_claim_keyed_file(tmp_path):
    from retrieval.store import KnowledgeStore, open_store

    path = pool_file(
        tmp_path, [{"claim_id": 0, "url": "https://a.org/1", "url2text": ["metro opened."] * 4}]
    )
    assert isinstance(open_store(path), KnowledgeStore)


def test_pooled_file_actually_returns_evidence_through_open_store(tmp_path):
    """The regression: KnowledgeStore returns nothing for a pooled file."""
    from retrieval.store import KnowledgeStore, open_store

    path = pool_file(
        tmp_path, [{"url": "https://a.org/1", "url2text": ["The Kochi metro opened in 2017."] * 4}]
    )
    assert KnowledgeStore(path).retrieve("0", "Kochi metro", k=5) == []
    assert open_store(path).retrieve("0", "Kochi metro", k=5) != []


def test_empty_file_does_not_crash_detection(tmp_path):
    from retrieval.store import open_store

    assert open_store(pool_file(tmp_path, [])).retrieve("0", "q", k=3) == []
