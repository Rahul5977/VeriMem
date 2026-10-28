import json
import random

import pytest

from attacks.memory import Target
from attacks.sources import (
    count_poisoned,
    fabricate_page,
    poison_pool,
    poisoned_urls,
)
from evaluator.score import LexicalScorer
from retrieval.store import PooledKnowledgeStore

DOMAINS = ("fakenews1.com", "fakenews2.org")


def target(
    cid="c1",
    claim="The health ministry reported 500 cases in 2020.",
    gold="Refuted",
    want="Supported",
):
    return Target(cid, claim, gold, want)


def clean_rows():
    return [
        {
            "url": "https://bbc.co.uk/a",
            "url2text": ["The health ministry published its figures for 2020."] * 6,
        },
        {
            "url": "https://nature.com/b",
            "url2text": ["Penguins are flightless birds of the southern hemisphere."] * 6,
        },
    ]


# --- the fabricated page ------------------------------------------------------------


def test_page_is_in_the_pool_format():
    p = fabricate_page(target(), random.Random(1), DOMAINS)
    assert set(p) >= {"url", "url2text"}
    assert isinstance(p["url2text"], list) and p["url2text"]


def test_page_weaves_in_the_claim_so_it_can_be_retrieved():
    t = target()
    p = fabricate_page(t, random.Random(1), DOMAINS)
    body = t.claim.rstrip(".").lower()
    assert any(body in s.lower() for s in p["url2text"])


def test_page_argues_for_the_adversary_verdict():
    p = fabricate_page(target(), random.Random(1), DOMAINS)
    text = " ".join(p["url2text"]).lower()
    assert "accurate and supported" in text  # the stance for "Supported"


def test_page_uses_a_misinformation_domain():
    p = fabricate_page(target(), random.Random(1), DOMAINS)
    assert any(d in p["url"] for d in DOMAINS)


def test_page_is_marked_for_bookkeeping():
    p = fabricate_page(target(), random.Random(1), DOMAINS)
    assert p["poisoned"] and p["target_claim_id"] == "c1"


def test_pages_for_the_same_target_differ():
    a = fabricate_page(target(), random.Random(1), DOMAINS, index=0)
    b = fabricate_page(target(), random.Random(1), DOMAINS, index=1)
    assert a["url"] != b["url"]


def test_page_length_is_respected():
    p = fabricate_page(target(), random.Random(1), DOMAINS, n_sentences=5)
    assert len(p["url2text"]) == 5


# --- injection ----------------------------------------------------------------------


def test_pool_gains_the_expected_number_of_pages():
    rows, stats = poison_pool(
        clean_rows(), [target("a"), target("b")], pages_per_target=3, domains=DOMAINS
    )
    assert stats["n_fake_pages"] == 6
    assert len(rows) == 2 + 6


def test_zero_pages_leaves_the_pool_alone():
    rows, stats = poison_pool(clean_rows(), [target()], pages_per_target=0, domains=DOMAINS)
    assert stats["n_fake_pages"] == 0 and len(rows) == 2


def test_real_pages_survive():
    rows, _ = poison_pool(clean_rows(), [target()], 3, domains=DOMAINS)
    assert sum(1 for r in rows if not r.get("poisoned")) == 2


def test_fakes_are_shuffled_in():
    rows, _ = poison_pool(clean_rows() * 20, [target()], 20, seed=4, domains=DOMAINS)
    idx = [i for i, r in enumerate(rows) if r.get("poisoned")]
    assert min(idx) < len(rows) // 3


def test_poisoning_needs_targets():
    with pytest.raises(ValueError, match="needs claims to attack"):
        poison_pool(clean_rows(), [], 3)


def test_negative_pages_rejected():
    with pytest.raises(ValueError, match="must be >= 0"):
        poison_pool(clean_rows(), [target()], -1)


# --- what the attack achieves -------------------------------------------------------


def test_fabricated_pages_are_actually_retrieved(tmp_path):
    """If the retriever never surfaces them the attack cannot work, and the study would
    be measuring retrieval rather than poisoning."""
    rows, _ = poison_pool(clean_rows(), [target()], pages_per_target=3, domains=DOMAINS)
    path = tmp_path / "pool.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    store = PooledKnowledgeStore(path)
    hits = store.retrieve("c1", target().claim, k=5)
    fake = poisoned_urls(rows)
    assert count_poisoned(hits, fake) > 0


def test_the_evaluator_scores_fabricated_pages_highly(tmp_path):
    """The uncomfortable part, and the reason C2 cannot defend this surface: a page
    written about the claim *is* relevant. Relevance is not truthfulness."""
    rows, _ = poison_pool(clean_rows(), [target()], pages_per_target=3, domains=DOMAINS)
    path = tmp_path / "pool.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    store = PooledKnowledgeStore(path)
    hits = store.retrieve("c1", target().claim, k=5)
    fake_urls = poisoned_urls(rows)
    scorer = LexicalScorer()
    fake_scores = [
        s
        for h, s in zip(hits, scorer.score(target().claim, [h.text for h in hits]), strict=True)
        if h.url in fake_urls
    ]
    assert fake_scores and max(fake_scores) > 0


def test_poisoned_urls_identifies_only_the_fakes():
    rows, _ = poison_pool(clean_rows(), [target()], 2, domains=DOMAINS)
    fake = poisoned_urls(rows)
    assert len(fake) == 2 and all(any(d in u for d in DOMAINS) for u in fake)
