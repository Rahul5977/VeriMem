"""Source poisoning end to end (P3.6, C1).

The claim this runner exists to test is that **C2 cannot defend the evidence path**: a
fabricated page is written about the target claim and is genuinely relevant, so the
evaluator scores it high and correctly. These tests pin the machinery that measures that
-- the in-memory corpus, the trust filter on strips, and the grade-separation statistic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attacks.memory import DEFAULT_DOMAINS, Target
from attacks.sources import count_poisoned, fabricate_page, poison_pool, poisoned_urls
from eval.run_source_poisoning import CONFIGS, grade_separation
from experience.format import Evidence, Step, Trajectory
from experience.store import BetaTrust
from retrieval.refine import Strip, TrustFilterStats, filter_by_trust
from retrieval.store import PooledKnowledgeStore

TARGETS = [
    Target("c1", "The Kochi metro opened to the public in 2017.", "Supported", "Refuted"),
    Target("c2", "Aspirin cures the common cold.", "Refuted", "Supported"),
]

REAL_ROWS = [
    {
        "url": "https://reuters.com/a",
        "url2text": [
            "The Kochi Metro began commercial operations in June 2017 after several delays.",
            "The line was built over six years at considerable public expense.",
        ],
    },
    {
        "url": "https://bbc.co.uk/b",
        "url2text": [
            "Medical reviews find no evidence that aspirin cures the common cold.",
            "It is widely used for pain relief and fever reduction instead.",
        ],
    },
]


def test_poisoning_adds_the_requested_pages_per_target() -> None:
    rows, stats = poison_pool(REAL_ROWS, TARGETS, pages_per_target=3, seed=1)
    assert stats["n_fake_pages"] == 6
    assert stats["n_real_pages"] == 2
    assert len(rows) == 8
    assert len(poisoned_urls(rows)) == 6


def test_zero_pages_leaves_the_pool_untouched() -> None:
    """The 0-page cell is the baseline every other cell is measured against."""
    rows, stats = poison_pool(REAL_ROWS, TARGETS, pages_per_target=0, seed=1)
    assert stats["n_fake_pages"] == 0
    assert poisoned_urls(rows) == set()
    assert len(rows) == len(REAL_ROWS)


def test_poisoning_with_no_targets_is_refused() -> None:
    with pytest.raises(ValueError, match="needs claims to attack"):
        poison_pool(REAL_ROWS, [], pages_per_target=3)


def test_the_poisoned_corpus_is_never_written_to_disk(tmp_path: Path) -> None:
    """Ethics rule 10. The corpus exists for one run and dies with the process."""
    pool = tmp_path / "pool.jsonl"
    pool.write_text("\n".join(json.dumps(r) for r in REAL_ROWS), encoding="utf-8")
    before = pool.read_text(encoding="utf-8")

    rows, _ = poison_pool(REAL_ROWS, TARGETS, pages_per_target=3, seed=1)
    store = PooledKnowledgeStore.from_rows(rows)
    assert len(store) > 0, "the in-memory store must actually index the rows"

    assert pool.read_text(encoding="utf-8") == before
    assert not any(p != pool for p in tmp_path.rglob("*.jsonl")), "no new file was written"


def test_a_fabricated_page_is_retrievable_for_its_target() -> None:
    """An attack that is never retrieved measures nothing, so this is a precondition."""
    rows, _ = poison_pool(REAL_ROWS, TARGETS, pages_per_target=3, seed=1)
    store = PooledKnowledgeStore.from_rows(rows)
    hits = store.retrieve("c1", TARGETS[0].claim, k=5)
    assert count_poisoned(hits, poisoned_urls(rows)) > 0


def test_trust_drops_strips_from_untrusted_publishers() -> None:
    trust = BetaTrust()
    for _ in range(8):
        trust.update("reuters.com", was_correct=True)

    strips = [
        Strip(
            text="a real sentence about the claim",
            url="https://reuters.com/a",
            domain="reuters.com",
            score=0.9,
        ),
        Strip(
            text="a fabricated sentence",
            url="https://channel23news.com/x",
            domain="channel23news.com",
            score=0.95,
        ),
    ]
    stats = TrustFilterStats()
    kept = filter_by_trust(
        strips, trust, floor=0.3, fake_urls={"https://channel23news.com/x"}, stats=stats
    )

    assert [s.domain for s in kept] == ["reuters.com"]
    assert stats.poisoned_considered == 1
    assert stats.poisoned_kept == 0
    assert stats.as_dict()["poison_pass_rate"] == 0.0


def test_trust_never_empties_the_evidence_block() -> None:
    """A verdict on no evidence is the attack succeeding by another route, not a defence."""
    strips = [
        Strip(text="one", url="https://a.example/1", domain="a.example", score=0.5),
        Strip(text="two", url="https://b.example/2", domain="b.example", score=0.9),
    ]
    kept = filter_by_trust(strips, BetaTrust(), floor=0.99)  # nothing can clear this
    assert len(kept) == 1
    assert kept[0].text == "two", "the fallback keeps the best available, not an arbitrary one"


def test_an_adaptive_source_attack_passes_the_trust_filter() -> None:
    """Impersonating a trusted publisher defeats C3, and C2 was never able to help."""
    trust = BetaTrust()
    for _ in range(8):
        trust.update("reuters.com", was_correct=True)

    fake = Strip(
        text="a fabricated sentence on a trusted-looking domain",
        url="https://reuters.com/fake",
        domain="reuters.com",
        score=0.95,
    )
    stats = TrustFilterStats()
    kept = filter_by_trust([fake], trust, floor=0.3, fake_urls={fake.url}, stats=stats)
    assert len(kept) == 1
    assert stats.poisoned_kept == 1


def _traj(urls_and_grades: list[tuple[str, float]]) -> Trajectory:
    return Trajectory(
        claim="a claim",
        verdict="Refuted",
        rationale="r",
        evidence=[Evidence(url=u, text="t", grade=g) for u, g in urls_and_grades],
        steps=[Step("retrieve", "top-5")],
        claim_id="c1",
        gold_label="Refuted",
    )


def test_grade_separation_reports_whether_the_evaluator_can_tell() -> None:
    """The empirical content of "C2 cannot defend this path"."""
    trajectories = [_traj([("https://real/1", 0.4), ("https://fake/1", 0.9)])]
    out = grade_separation(trajectories, {"https://fake/1"})
    assert out["mean_grade_real"] == 0.4
    assert out["mean_grade_fake"] == 0.9
    # Negative separation means the fakes scored HIGHER than the truth.
    assert out["separation"] == pytest.approx(-0.5)


def test_grade_separation_is_absent_rather_than_zero_with_no_fakes() -> None:
    """A 0-page cell has nothing to compare, and reporting 0.0 would read as "no effect"."""
    out = grade_separation([_traj([("https://real/1", 0.4)])], set())
    assert out.get("separation") is None
    assert out["mean_grade_fake"] is None


def test_the_configs_differ_only_in_trust() -> None:
    assert CONFIGS == {"undefended": False, "source_trust": True}


def test_a_fabricated_page_reads_like_a_page() -> None:
    """A slogan would be dismissed for the wrong reason and the study would measure prose."""
    import random

    page = fabricate_page(TARGETS[0], random.Random(0), DEFAULT_DOMAINS, 0, n_sentences=12)
    assert len(page["url2text"]) == 12
    assert page["poisoned"] is True
    assert page["target_claim_id"] == "c1"
    # The claim's own wording has to be present or BM25 will never surface it.
    assert "Kochi" in " ".join(page["url2text"])
