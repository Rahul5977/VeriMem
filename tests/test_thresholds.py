import pytest

from evaluator.thresholds import BRANCHES, Outcome, branch_only, sweep


def outcome(cid, gold, grade, correct, incorrect, ambiguous):
    return Outcome(
        cid, gold, grade, {"correct": correct, "incorrect": incorrect, "ambiguous": ambiguous}
    )


# --- picking a branch ---------------------------------------------------------------


def test_high_grade_takes_the_correct_branch():
    o = outcome("c1", "Supported", 0.8, "Supported", "Refuted", "Refuted")
    assert o.verdict_for(upper=0.5, lower=-0.5) == "Supported"
    assert o.action_for(upper=0.5, lower=-0.5) == "correct"


def test_low_grade_takes_the_incorrect_branch():
    o = outcome("c1", "Supported", -0.8, "Refuted", "Supported", "Refuted")
    assert o.verdict_for(0.5, -0.5) == "Supported"
    assert o.action_for(0.5, -0.5) == "incorrect"


def test_middle_grade_is_ambiguous():
    o = outcome("c1", "Supported", 0.0, "Refuted", "Refuted", "Supported")
    assert o.verdict_for(0.5, -0.5) == "Supported"
    assert o.action_for(0.5, -0.5) == "ambiguous"


def test_boundaries_are_inclusive():
    o = outcome("c1", "Supported", 0.5, "A", "B", "C")
    assert o.action_for(0.5, -0.5) == "correct"
    assert outcome("c2", "Supported", -0.5, "A", "B", "C").action_for(0.5, -0.5) == "incorrect"


# --- the sweep ----------------------------------------------------------------------


def two_claims():
    # grade 0.9 -> only "correct" answers rightly; grade -0.9 -> only "incorrect" does
    return [
        outcome("a", "Supported", 0.9, "Supported", "Refuted", "Refuted"),
        outcome("b", "Refuted", -0.9, "Supported", "Refuted", "Supported"),
    ]


def test_sweep_finds_the_pair_that_routes_both_claims_correctly():
    rows = sweep(two_claims(), [0.5, 0.95], [-0.5, -0.95])
    assert rows[0]["macro_f1"] == 1.0
    assert rows[0]["upper"] == 0.5 and rows[0]["lower"] == -0.5


def test_sweep_is_sorted_by_macro_f1():
    rows = sweep(two_claims(), [0.0, 0.5, 0.95], [-0.95, -0.5, 0.0])
    assert rows == sorted(rows, key=lambda r: -r["macro_f1"])


def test_sweep_skips_pairs_where_lower_is_not_below_upper():
    rows = sweep(two_claims(), [0.0], [0.0, 0.5])
    assert all(r["lower"] < r["upper"] for r in rows)
    assert rows == [] or len(rows) == 0 or all(r["upper"] == 0.0 for r in rows)


def test_sweep_reports_action_shares_that_sum_to_one():
    rows = sweep(two_claims(), [0.5], [-0.5])
    assert sum(rows[0]["action_share"].values()) == pytest.approx(1.0, abs=0.01)


def test_sweep_ignores_unlabelled_claims():
    rows = sweep([outcome("x", None, 0.5, "A", "B", "C"), *two_claims()], [0.5], [-0.5])
    assert rows[0]["macro_f1"] == 1.0


def test_empty_grid_returns_nothing():
    assert sweep(two_claims(), [0.0], [0.5]) == []


# --- the floor ----------------------------------------------------------------------


def test_always_one_branch_is_computed_for_each():
    outcomes = two_claims()
    floors = {b: branch_only(outcomes, b) for b in BRANCHES}
    assert set(floors) == set(BRANCHES)
    # "correct" gets a right and a wrong answer here, so it cannot be perfect
    assert floors["correct"]["accuracy"] == 0.5


def test_routing_can_beat_every_single_branch():
    """The whole justification for routing: if always taking one branch matched the
    best routed result, the router would be dead weight."""
    outcomes = two_claims()
    best_single = max(branch_only(outcomes, b)["macro_f1"] for b in BRANCHES)
    best_routed = sweep(outcomes, [0.5], [-0.5])[0]["macro_f1"]
    assert best_routed > best_single
