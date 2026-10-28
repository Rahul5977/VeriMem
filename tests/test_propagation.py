"""Propagation pairs and the conditions that make the result attributable (P3.8, C1).

The finding this supports is that a poisoned verdict keeps doing harm after every
fabricated record is deleted, because the record that carries it forward is the
pipeline's own genuine output. The risk is claiming that from noise, so most of these
tests are about the controls rather than the attack.
"""

from __future__ import annotations

from pathlib import Path

from attacks.memory import Target
from attacks.propagation import Pair, choose_pairs, corrupted_copy, read_pairs, write_pairs
from core.config import RunConfig
from eval.datasets import Claim
from eval.run_propagation import CONDITIONS, PairOutcome, store_for
from experience.format import Evidence, Step, Trajectory

CLAIMS = [
    Claim(id="a1", claim="The Kochi metro opened to the public in 2017.", gold_label="Supported"),
    Claim(
        id="b1", claim="The Kochi metro cost more than its original budget.", gold_label="Refuted"
    ),
    Claim(
        id="c1", claim="Penguins live mainly in the southern hemisphere.", gold_label="Supported"
    ),
    Claim(id="d1", claim="Aspirin cures the common cold.", gold_label="Refuted"),
]
TARGETS = [Target("a1", CLAIMS[0].claim, "Supported", "Refuted")]


def _traj(
    claim_id: str, claim: str, verdict: str, gold: str, domain: str = "reuters.com"
) -> Trajectory:
    return Trajectory(
        claim=claim,
        verdict=verdict,
        rationale=f"The evidence indicates {verdict.lower()}.",
        evidence=[Evidence(url=f"https://{domain}/{claim_id}", text="a real retrieved passage")],
        steps=[Step("retrieve", "top-5")],
        claim_id=claim_id,
        gold_label=gold,
    )


# --- pairing -------------------------------------------------------------------------


def test_a_target_is_paired_with_a_lexically_close_claim() -> None:
    pairs = choose_pairs(CLAIMS, TARGETS, min_similarity=0.1, per_target=1)
    assert [p.neighbour_id for p in pairs] == ["b1"]
    assert 0 < pairs[0].similarity <= 1.0, "similarity is a fraction of the self-score"


def test_a_target_is_never_paired_with_itself() -> None:
    """A's own verdict reaching A is direct harm, not propagation."""
    pairs = choose_pairs(CLAIMS, TARGETS, min_similarity=0.0, per_target=4)
    assert all(p.neighbour_id != p.target_id for p in pairs)


def test_two_targets_are_never_paired_with_each_other() -> None:
    """A claim under direct attack cannot demonstrate propagated harm."""
    both = [
        Target("a1", CLAIMS[0].claim, "Supported", "Refuted"),
        Target("b1", CLAIMS[1].claim, "Refuted", "Supported"),
    ]
    pairs = choose_pairs(CLAIMS, both, min_similarity=0.0, per_target=3)
    ids = {"a1", "b1"}
    assert all(p.neighbour_id not in ids for p in pairs)


def test_unlabelled_neighbours_are_not_paired() -> None:
    """Without a gold label there is no way to show B became *wrong*."""
    claims = [*CLAIMS, Claim(id="e1", claim="The Kochi metro is in Kerala.", gold_label=None)]
    pairs = choose_pairs(claims, TARGETS, min_similarity=0.0, per_target=5)
    assert all(p.neighbour_id != "e1" for p in pairs)
    assert all(p.neighbour_gold for p in pairs)


def test_similarity_is_relative_so_the_threshold_survives_a_bigger_corpus() -> None:
    """An absolute BM25 cutoff finds nothing on a small set and everything on a big one."""
    small = choose_pairs(CLAIMS, TARGETS, min_similarity=0.1, per_target=1)
    padded = CLAIMS + [
        Claim(
            id=f"pad{i}",
            claim=f"An unrelated statement number {i} about nothing.",
            gold_label="Refuted",
        )
        for i in range(60)
    ]
    big = choose_pairs(padded, TARGETS, min_similarity=0.1, per_target=1)
    assert [p.neighbour_id for p in small] == [p.neighbour_id for p in big] == ["b1"]


def test_an_unreachable_neighbour_is_excluded() -> None:
    """No retrieval channel, nothing to propagate along."""
    isolated = [CLAIMS[0], CLAIMS[2]]  # metro vs penguins
    assert choose_pairs(isolated, TARGETS, min_similarity=0.5, per_target=1) == []


def test_pairs_round_trip_through_disk(tmp_path: Path) -> None:
    pairs = choose_pairs(CLAIMS, TARGETS, min_similarity=0.1, per_target=1)
    path = write_pairs(pairs, tmp_path / "pairs.jsonl")
    assert read_pairs(path) == pairs


# --- the corrupted record ------------------------------------------------------------


def test_corruption_changes_the_verdict_and_nothing_else() -> None:
    """Control and treatment must differ in one label, or a shift is not attributable."""
    original = _traj("a1", CLAIMS[0].claim, "Supported", "Supported")
    corrupted = corrupted_copy(original, "Refuted")

    assert corrupted.verdict == "Refuted"
    assert corrupted.claim == original.claim
    assert [e.url for e in corrupted.evidence] == [e.url for e in original.evidence]
    assert [e.text for e in corrupted.evidence] == [e.text for e in original.evidence]
    assert corrupted.metadata["original_verdict"] == "Supported"


def test_the_corrupted_record_does_not_declare_itself_poisoned() -> None:
    """It is the pipeline's own honest output. Flagging it would let bookkeeping find
    what the defences provably cannot, and the whole point is that they cannot."""
    corrupted = corrupted_copy(_traj("a1", CLAIMS[0].claim, "Supported", "Supported"), "Refuted")
    assert corrupted.poisoned is False
    assert corrupted.correct is False, "it is still wrong against gold, just not fabricated"


def test_the_corrupted_record_cites_ordinary_publishers() -> None:
    """C3 has nothing to object to: the fakes that misled A are no longer in the picture."""
    corrupted = corrupted_copy(_traj("a1", CLAIMS[0].claim, "Supported", "Supported"), "Refuted")
    assert all("reuters.com" in d for d in corrupted.domains)


# --- the three conditions ------------------------------------------------------------


def test_the_three_stores_differ_only_in_a() -> None:
    clean = [
        _traj("x1", "An unrelated earlier verification.", "Refuted", "Refuted"),
        _traj("a1", CLAIMS[0].claim, "Supported", "Supported"),
    ]
    a = clean[1]
    stores = {
        c: store_for(c, clean, a, "Refuted", scorer=None, trust=None, cfg=RunConfig())
        for c in CONDITIONS
    }
    assert len(stores["absent"]) == 1
    assert len(stores["clean"]) == len(stores["corrupted"]) == 2

    verdicts = {c: {t.claim_id: t.verdict for t in s.trajectories} for c, s in stores.items()}
    assert verdicts["clean"]["a1"] == "Supported"
    assert verdicts["corrupted"]["a1"] == "Refuted"
    # Everything that is not A is identical across all three.
    for c in CONDITIONS:
        assert verdicts[c]["x1"] == "Refuted"


def test_a_stale_copy_of_a_is_removed_before_the_condition_is_built() -> None:
    """The clean history already contains A. Leaving it in would put two A records in the
    corrupted store, one of them carrying the right answer."""
    clean = [_traj("a1", CLAIMS[0].claim, "Supported", "Supported")]
    store = store_for("corrupted", clean, clean[0], "Refuted", None, None, RunConfig())
    assert len(store) == 1
    assert store.trajectories[0].verdict == "Refuted"


# --- what counts as propagation ------------------------------------------------------


def _outcome(**kw) -> PairOutcome:
    pair = Pair("a1", CLAIMS[0].claim, "b1", CLAIMS[1].claim, "Refuted", 0.3)
    base = {
        "pair": pair,
        "target_verdict_under_attack": "Refuted",
        "target_was_flipped": True,
        "verdicts": {"absent": "Refuted", "clean": "Refuted", "corrupted": "Refuted"},
        "retrieved_a": {"absent": False, "clean": True, "corrupted": True},
    }
    return PairOutcome(**{**base, **kw})


def test_propagation_requires_a_change_against_the_clean_control() -> None:
    """B already answering the adversary's verdict without A is not propagation."""
    o = _outcome(verdicts={"absent": "Refuted", "clean": "Refuted", "corrupted": "Refuted"})
    assert not o.propagated


def test_propagation_is_counted_when_only_the_corrupted_store_moves_b() -> None:
    o = _outcome(verdicts={"absent": "Supported", "clean": "Supported", "corrupted": "Refuted"})
    assert o.propagated


def test_propagation_requires_a_to_have_actually_been_retrieved() -> None:
    """Otherwise pipeline nondeterminism is counted as the attack spreading."""
    o = _outcome(
        verdicts={"absent": "Supported", "clean": "Supported", "corrupted": "Refuted"},
        retrieved_a={"absent": False, "clean": True, "corrupted": False},
    )
    assert not o.propagated


def test_nothing_propagates_from_a_target_the_attack_did_not_flip() -> None:
    """No corrupted verdict was produced, so there is nothing to carry forward."""
    o = _outcome(
        target_was_flipped=False,
        verdicts={"absent": "Supported", "clean": "Supported", "corrupted": "Refuted"},
    )
    assert not o.propagated
    assert not o.collateral


def test_collateral_is_b_going_from_right_to_wrong() -> None:
    o = _outcome(verdicts={"absent": "Refuted", "clean": "Refuted", "corrupted": "Supported"})
    assert o.collateral, "clean matched gold Refuted, corrupted did not"


def test_collateral_is_not_counted_when_b_was_already_wrong() -> None:
    """B that the clean store also gets wrong was not harmed by the poison."""
    o = _outcome(verdicts={"absent": "Supported", "clean": "Supported", "corrupted": "Supported"})
    assert not o.collateral


def test_the_outcome_serialises_its_own_verdict() -> None:
    d = _outcome().as_dict()
    assert d["target_id"] == "a1"
    assert set(d["verdicts"]) == set(CONDITIONS)
    assert isinstance(d["propagated"], bool)
    assert isinstance(d["collateral"], bool)


# --- duplicates are not neighbours ----------------------------------------------------


def test_an_identical_claim_is_not_a_propagation_neighbour() -> None:
    """AVeriTeC dev really contains this: claims 192 and 208 are byte-identical under
    different ids. A verdict reaching its own restatement has propagated nowhere."""
    claims = [
        CLAIMS[0],
        Claim(id="dup", claim=CLAIMS[0].claim, gold_label="Supported"),
        CLAIMS[1],
    ]
    pairs = choose_pairs(claims, TARGETS, min_similarity=0.0, per_target=1)
    assert all(p.neighbour_id != "dup" for p in pairs)
    assert [p.neighbour_id for p in pairs] == ["b1"], "it falls through to a real neighbour"


def test_a_claim_differing_only_in_punctuation_is_still_a_duplicate() -> None:
    """Claims 194 and 224 differ by one trailing full stop."""
    claims = [
        CLAIMS[0],
        Claim(id="dup", claim=CLAIMS[0].claim.rstrip(".") + "  ", gold_label="Supported"),
        CLAIMS[1],
    ]
    pairs = choose_pairs(claims, TARGETS, min_similarity=0.0, per_target=1)
    assert all(p.neighbour_id != "dup" for p in pairs)


def test_a_near_restatement_is_excluded_by_the_similarity_ceiling() -> None:
    """Not textually identical, but close enough that it is the same assertion."""
    near = Claim(
        id="near",
        claim="The Kochi metro opened to the public in 2017",  # no full stop, same words
        gold_label="Supported",
    )
    pairs = choose_pairs([CLAIMS[0], near, CLAIMS[1]], TARGETS, min_similarity=0.0, per_target=2)
    assert all(p.similarity <= 0.9 for p in pairs)
    assert all(p.neighbour_id != "near" for p in pairs)


def test_a_genuine_neighbour_survives_the_ceiling() -> None:
    """The guard must not throw away the pairs the experiment is made of."""
    pairs = choose_pairs(CLAIMS, TARGETS, min_similarity=0.1, per_target=1)
    assert [p.neighbour_id for p in pairs] == ["b1"]
    assert 0.1 <= pairs[0].similarity <= 0.9
