"""Training toward the gold label rather than the agent's verdict (P4, C4).

The first SFT runs kept only trajectories whose verdict matched gold, which is the
obvious reading of "don't train on wrong answers". It starved the minority classes: of
885 examples, 5 were "Conflicting evidence" -- 0.6% -- because those are exactly the
claims CRAG gets wrong, so they were the ones filtered out. A model trained on that could
never learn the class, and the failure compounds with every round of self-training.

`--gold-targets` keeps every labelled trajectory and teaches the true label instead. The
(claim, evidence) pair is still real work the agent did; only the answer is corrected.
"""

from __future__ import annotations

from experience.format import Evidence, Step, Trajectory
from training.data import build_dataset, build_example, target_verdict


def _answer(example) -> str:
    """The text the model is trained to produce."""
    return example.completion[-1]["content"]


def _traj(verdict: str, gold: str, i: int = 0) -> Trajectory:
    return Trajectory(
        claim=f"A claim numbered {i} about a measurable fact.",
        verdict=verdict,
        rationale=f"The evidence indicates {verdict.lower()}.",
        evidence=[Evidence(url=f"https://example.com/{i}", text="Some retrieved passage.")],
        steps=[Step("retrieve", "top-5"), Step("verify", "verdict produced")],
        claim_id=f"c{i}",
        gold_label=gold,
    )


def test_gold_target_overrides_a_wrong_verdict() -> None:
    wrong = _traj("Refuted", "Conflicting evidence")
    assert target_verdict(wrong, use_gold=False) == "Refuted"
    assert target_verdict(wrong, use_gold=True) == "Conflicting evidence"


def test_gold_target_is_a_no_op_when_the_agent_was_right() -> None:
    right = _traj("Supported", "Supported")
    assert target_verdict(right, use_gold=True) == "Supported"
    assert _answer(build_example(right, use_gold=True)) == right.render_answer()


def test_an_unlabelled_trajectory_falls_back_to_its_own_verdict() -> None:
    """Poisoned and live-demo trajectories carry no gold label to train toward."""
    unlabelled = _traj("Supported", "")
    unlabelled.gold_label = None
    assert target_verdict(unlabelled, use_gold=True) == "Supported"


def test_a_corrected_answer_drops_the_rationale_that_argued_the_other_way() -> None:
    """A rationale for "Refuted" beside a "Supported" target teaches self-contradiction."""
    answer = _answer(build_example(_traj("Refuted", "Supported"), use_gold=True))
    assert "Verdict: Supported" in answer
    assert "refuted" not in answer.lower()


def test_gold_targets_keep_the_failures_that_filtering_discards() -> None:
    trajectories = [
        _traj("Refuted", "Refuted", 0),
        _traj("Refuted", "Conflicting evidence", 1),
        _traj("Supported", "Not enough evidence", 2),
    ]

    filtered, fstats = build_dataset(trajectories, mode="plain")
    gold, gstats = build_dataset(trajectories, mode="plain", use_gold=True)

    assert len(filtered) == 1, "only the correct verification survives filtering"
    assert fstats.n_skipped == 2
    assert len(gold) == 3, "every labelled trajectory is usable with gold targets"
    assert gstats.n_skipped == 0


def test_gold_targets_restore_the_minority_classes() -> None:
    """The actual bug: filtering reported a distribution the data did not have."""
    trajectories = [_traj("Refuted", "Refuted", i) for i in range(10)]
    trajectories += [_traj("Refuted", "Conflicting evidence", 10 + i) for i in range(4)]

    _, filtered = build_dataset(trajectories, mode="plain")
    _, gold = build_dataset(trajectories, mode="plain", use_gold=True)

    assert filtered.verdict_counts.get("Conflicting evidence", 0) == 0
    assert gold.verdict_counts["Conflicting evidence"] == 4
    assert gold.verdict_counts["Refuted"] == 10


def test_verdict_counts_report_what_is_taught_not_what_was_predicted() -> None:
    """A stats block that counted agent verdicts would have hidden the whole problem."""
    _, stats = build_dataset([_traj("Refuted", "Supported")], mode="plain", use_gold=True)
    assert stats.verdict_counts == {"Supported": 1}
