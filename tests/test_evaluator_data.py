import json

import pytest

from eval.datasets import Claim
from evaluator.build_data import (
    EASY_NEGATIVE,
    HARD_NEGATIVE,
    POSITIVE,
    build_pairs,
    gold_urls,
    split_by_claim,
)
from evaluator.build_data import main as build_main
from retrieval.store import PooledKnowledgeStore

GOLD = "https://gold.example.org/a"
OTHER = "https://other.example.org/b"
FAR = "https://unrelated.example.org/c"


def pool_file(tmp_path, rows):
    path = tmp_path / "pool.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def make_pool(tmp_path):
    return pool_file(
        tmp_path,
        [
            {"url": GOLD, "url2text": ["The Kochi metro opened in June 2017 officially."] * 6},
            {"url": OTHER, "url2text": ["The Kochi metro extension was discussed in 2019."] * 6},
            {"url": FAR, "url2text": ["Penguins are flightless birds of the south."] * 6},
        ],
    )


def claim_with_gold(url=GOLD, claim_id="c1", text="When did the Kochi metro open?"):
    return Claim(
        id=claim_id,
        claim=text,
        questions=[{"question": "q", "answers": [{"answer": "a", "source_url": url}]}],
    )


# --- gold URL extraction ------------------------------------------------------------


def test_collects_gold_urls_from_every_answer():
    c = Claim(
        id="c1",
        claim="x",
        questions=[
            {"answers": [{"source_url": "https://a.org/1"}]},
            {"answers": [{"cached_source_url": "https://b.org/2"}]},
        ],
    )
    assert gold_urls(c) == {"https://a.org/1", "https://b.org/2"}


def test_no_answers_means_no_gold():
    assert gold_urls(Claim(id="c", claim="x")) == set()


# --- pair construction --------------------------------------------------------------


def test_positives_come_from_the_cited_url(tmp_path):
    store = PooledKnowledgeStore(make_pool(tmp_path))
    pairs, _ = build_pairs([claim_with_gold()], store)
    positives = [p for p in pairs if p.kind == POSITIVE]
    assert positives and all(p.url == GOLD and p.label == 1 for p in positives)


def test_hard_negatives_are_retrieved_but_not_cited(tmp_path):
    store = PooledKnowledgeStore(make_pool(tmp_path))
    pairs, _ = build_pairs([claim_with_gold()], store, hard_per_claim=2)
    hard = [p for p in pairs if p.kind == HARD_NEGATIVE]
    assert hard, "the lexically similar non-gold page should surface"
    assert all(p.url != GOLD and p.label == 0 for p in hard)
    # the similar page, not the penguin one, is what BM25 should rank
    assert any(p.url == OTHER for p in hard)


def test_a_cited_url_is_never_a_negative(tmp_path):
    store = PooledKnowledgeStore(make_pool(tmp_path))
    pairs, _ = build_pairs([claim_with_gold()], store, hard_per_claim=10, easy_per_claim=10)
    assert all(p.label == 1 for p in pairs if p.url == GOLD)


def test_easy_negatives_are_produced(tmp_path):
    store = PooledKnowledgeStore(make_pool(tmp_path))
    _, stats = build_pairs([claim_with_gold()], store, easy_per_claim=2)
    assert stats[EASY_NEGATIVE] >= 1


def test_claims_whose_gold_pages_are_missing_are_skipped(tmp_path):
    store = PooledKnowledgeStore(make_pool(tmp_path))
    orphan = claim_with_gold(url="https://never-fetched.example.org/x", claim_id="c9")
    pairs, stats = build_pairs([orphan], store)
    assert pairs == [] and stats["claims_without_gold_in_pool"] == 1


def test_positives_are_capped(tmp_path):
    store = PooledKnowledgeStore(make_pool(tmp_path))
    pairs, _ = build_pairs([claim_with_gold()], store, max_positives=1)
    assert sum(1 for p in pairs if p.kind == POSITIVE) == 1


def test_is_deterministic_for_a_seed(tmp_path):
    store = PooledKnowledgeStore(make_pool(tmp_path))
    a, _ = build_pairs([claim_with_gold()], store, seed=7)
    b, _ = build_pairs([claim_with_gold()], store, seed=7)
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]


# --- the split ----------------------------------------------------------------------


def test_split_never_puts_a_claim_on_both_sides(tmp_path):
    store = PooledKnowledgeStore(make_pool(tmp_path))
    claims = [claim_with_gold(claim_id=f"c{i}", text=f"metro question {i}") for i in range(20)]
    pairs, _ = build_pairs(claims, store)
    train, dev = split_by_claim(pairs)
    assert {p.claim_id for p in train}.isdisjoint({p.claim_id for p in dev})


def test_split_keeps_every_pair(tmp_path):
    store = PooledKnowledgeStore(make_pool(tmp_path))
    claims = [claim_with_gold(claim_id=f"c{i}", text=f"metro question {i}") for i in range(20)]
    pairs, _ = build_pairs(claims, store)
    train, dev = split_by_claim(pairs)
    assert len(train) + len(dev) == len(pairs)


def test_split_marks_the_side_on_each_pair(tmp_path):
    store = PooledKnowledgeStore(make_pool(tmp_path))
    claims = [claim_with_gold(claim_id=f"c{i}", text=f"q {i}") for i in range(20)]
    train, dev = split_by_claim(build_pairs(claims, store)[0])
    assert all(p.split == "train" for p in train)
    assert all(p.split == "dev" for p in dev)


def test_dev_side_is_non_empty_even_for_few_claims(tmp_path):
    store = PooledKnowledgeStore(make_pool(tmp_path))
    pairs, _ = build_pairs([claim_with_gold(claim_id="c1")], store)
    _, dev = split_by_claim(pairs)
    assert dev


# --- the CLI ------------------------------------------------------------------------


def test_cli_refuses_a_pool_that_cannot_label_the_claims(tmp_path, capsys):
    claims_file = tmp_path / "train.json"
    claims_file.write_text(
        json.dumps(
            [
                {
                    "claim": "x",
                    "label": "Supported",
                    "questions": [{"answers": [{"source_url": "https://absent.example/1"}]}],
                }
            ]
        )
    )
    with pytest.raises(SystemExit, match="No pairs built"):
        build_main(
            [
                "--data",
                str(claims_file),
                "--pool",
                str(make_pool(tmp_path)),
                "--out",
                str(tmp_path / "out"),
            ]
        )


def test_cli_warns_when_given_the_dev_split(tmp_path, capsys):
    """Rule 5: an evaluator fitted on dev has seen the system's test set."""
    dev_file = tmp_path / "dev.json"
    dev_file.write_text(
        json.dumps(
            [
                {
                    "claim": "When did the Kochi metro open?",
                    "label": "Supported",
                    "questions": [{"answers": [{"source_url": GOLD}]}],
                }
            ]
        )
    )
    build_main(
        [
            "--data",
            str(dev_file),
            "--pool",
            str(make_pool(tmp_path)),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert "leaks" in capsys.readouterr().out


def test_cli_writes_both_files(tmp_path):
    train_file = tmp_path / "train.json"
    train_file.write_text(
        json.dumps(
            [
                {
                    "claim": f"When did the Kochi metro open {i}?",
                    "label": "Supported",
                    "questions": [{"answers": [{"source_url": GOLD}]}],
                }
                for i in range(12)
            ]
        )
    )
    out = tmp_path / "out"
    build_main(["--data", str(train_file), "--pool", str(make_pool(tmp_path)), "--out", str(out)])
    assert (out / "relevance_train.jsonl").exists()
    assert (out / "relevance_dev.jsonl").exists()
    rows = [json.loads(x) for x in (out / "relevance_train.jsonl").read_text().splitlines()]
    assert {"claim", "passage", "url", "label", "kind", "claim_id", "split"} == set(rows[0])


def test_positives_are_the_on_topic_part_of_the_cited_page(tmp_path):
    """Not the first passages, which are masthead and navigation.

    Hard negatives are BM25-selected for topical overlap. If positives are taken by
    position instead, the data pairs boilerplate-as-relevant against on-topic-text-as-
    irrelevant and teaches the inverse of the task.
    """
    path = pool_file(
        tmp_path,
        [
            {
                "url": GOLD,
                "url2text": (
                    ["Home Menu Subscribe Sign in Newsletter Contact us advertisement."] * 4
                    + ["The Kochi metro opened to the public in June 2017 after delays."] * 4
                ),
            }
        ],
    )
    store = PooledKnowledgeStore(path)
    pairs, _ = build_pairs([claim_with_gold()], store, max_positives=1, hard_per_claim=0)
    positive = next(p for p in pairs if p.kind == POSITIVE)
    assert "metro opened" in positive.passage, positive.passage
    assert "Subscribe" not in positive.passage
