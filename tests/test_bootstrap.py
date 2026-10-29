"""Confidence intervals for the report's tables.

This exists because of a specific near-miss. CRAG measured 0.414 macro-F1 in one run and
0.363 in another, and the G1 gate had been recorded as "CRAG 0.414 vs plain RAG 0.346" --
a 0.068 win. The paired bootstrap puts that difference at +0.009 [-0.032, 0.052]: the
ordering was never established. Macro-F1 on AVeriTeC dev weights four classes equally
while two of them hold ~36 instances each, so a quarter of the metric rests on one or two
correct predictions.

The tests below pin the properties that make an interval trustworthy, and in particular
the distinction between a marginal interval and a paired one -- getting that wrong is how
a real difference gets reported as "no effect", or a null gets reported as a win.
"""

from __future__ import annotations

import pytest

from eval.metrics import (
    bootstrap_difference,
    bootstrap_metric,
    pairs_from_confusion,
    proportion_interval,
    score,
)

LABELS = ("Supported", "Refuted", "Conflicting evidence", "Not enough evidence")


def _perfect(n: int = 200) -> tuple[list[str], list[str]]:
    gold = [LABELS[i % 4] for i in range(n)]
    return gold, list(gold)


# --- basic properties -----------------------------------------------------------------


def test_the_point_estimate_is_the_plain_metric() -> None:
    """The interval must be *around* the number that goes in the table, not near it."""
    gold, pred = _perfect()
    iv = bootstrap_metric(gold, pred, n_resamples=200)
    assert iv.point == pytest.approx(score(gold, pred).macro_f1)


def test_the_interval_brackets_the_point() -> None:
    gold = [LABELS[i % 4] for i in range(200)]
    pred = [LABELS[(i + (i % 3 == 0)) % 4] for i in range(200)]
    iv = bootstrap_metric(gold, pred, n_resamples=500)
    assert iv.low <= iv.point <= iv.high
    assert iv.width > 0


def test_a_perfect_system_has_a_degenerate_interval() -> None:
    """Every resample of an all-correct run is also all-correct."""
    gold, pred = _perfect()
    iv = bootstrap_metric(gold, pred, n_resamples=300)
    assert iv.point == pytest.approx(1.0)
    assert iv.width == pytest.approx(0.0, abs=1e-9)


def test_more_claims_give_a_narrower_interval() -> None:
    """The property that makes the whole exercise meaningful."""
    small_gold = [LABELS[i % 4] for i in range(80)]
    small_pred = [LABELS[(i + (i % 3 == 0)) % 4] for i in range(80)]
    big_gold = [LABELS[i % 4] for i in range(800)]
    big_pred = [LABELS[(i + (i % 3 == 0)) % 4] for i in range(800)]

    small = bootstrap_metric(small_gold, small_pred, n_resamples=600)
    big = bootstrap_metric(big_gold, big_pred, n_resamples=600)
    assert big.width < small.width


def test_it_is_deterministic_for_a_given_seed() -> None:
    """A number in a paper must not change when the table is regenerated."""
    gold = [LABELS[i % 4] for i in range(150)]
    pred = [LABELS[(i + 1) % 4] if i % 5 == 0 else gold[i] for i in range(150)]
    a = bootstrap_metric(gold, pred, n_resamples=400, seed=13)
    b = bootstrap_metric(gold, pred, n_resamples=400, seed=13)
    c = bootstrap_metric(gold, pred, n_resamples=400, seed=14)
    assert (a.low, a.high) == (b.low, b.high)
    assert (a.low, a.high) != (c.low, c.high), "different seeds should resample differently"


def test_accuracy_can_be_bootstrapped_too() -> None:
    gold = [LABELS[i % 4] for i in range(200)]
    pred = [LABELS[(i + 1) % 4] if i % 4 == 0 else gold[i] for i in range(200)]
    iv = bootstrap_metric(gold, pred, metric="accuracy", n_resamples=300)
    assert iv.point == pytest.approx(score(gold, pred).accuracy)


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="gold labels but"):
        bootstrap_metric(["Refuted"], ["Refuted", "Supported"])


def test_an_empty_evaluation_is_refused() -> None:
    with pytest.raises(ValueError, match="nothing to bootstrap"):
        bootstrap_metric([], [])


# --- paired vs marginal ---------------------------------------------------------------


def test_a_paired_difference_is_tighter_than_comparing_two_intervals() -> None:
    """Why `bootstrap_difference` exists.

    Two systems scored on one claim set fail on the same hard claims, so their
    difference is estimated far more precisely than either score. Judging by whether
    marginal intervals overlap would call a reliable difference "no effect".
    """
    gold = [LABELS[i % 4] for i in range(400)]
    # B is A with a few extra errors: strongly correlated, consistently a bit worse.
    pred_a = [LABELS[(i + 1) % 4] if i % 7 == 0 else gold[i] for i in range(400)]
    pred_b = [LABELS[(i + 1) % 4] if i % 7 == 0 or i % 23 == 0 else gold[i] for i in range(400)]

    a = bootstrap_metric(gold, pred_a, n_resamples=800)
    b = bootstrap_metric(gold, pred_b, n_resamples=800)
    diff = bootstrap_difference(gold, pred_a, pred_b, n_resamples=800)

    assert diff.point > 0
    assert diff.width < min(a.width, b.width), "the paired interval should be the tighter one"
    assert diff.low > 0, "a consistent advantage should exclude zero"
    # And the marginal intervals do overlap, which is the trap.
    assert a.low < b.high


def test_a_difference_of_zero_includes_zero() -> None:
    gold = [LABELS[i % 4] for i in range(200)]
    pred = [LABELS[(i + 1) % 4] if i % 6 == 0 else gold[i] for i in range(200)]
    diff = bootstrap_difference(gold, pred, list(pred), n_resamples=400)
    assert diff.point == pytest.approx(0.0)
    assert diff.low <= 0 <= diff.high


def test_the_difference_is_antisymmetric() -> None:
    gold = [LABELS[i % 4] for i in range(200)]
    a = [LABELS[(i + 1) % 4] if i % 7 == 0 else gold[i] for i in range(200)]
    b = [LABELS[(i + 1) % 4] if i % 4 == 0 else gold[i] for i in range(200)]
    ab = bootstrap_difference(gold, a, b, n_resamples=400)
    ba = bootstrap_difference(gold, b, a, n_resamples=400)
    assert ab.point == pytest.approx(-ba.point)


def test_a_paired_comparison_requires_the_same_claims() -> None:
    """Passing two systems scored on different claim sets is a silent lie, not an error
    to discover from a strange interval."""
    with pytest.raises(ValueError, match="same claims"):
        bootstrap_difference(["Refuted"] * 3, ["Refuted"] * 3, ["Refuted"] * 2)


# --- recovering pairs from a confusion matrix -----------------------------------------


def test_confusion_expansion_reproduces_the_metrics() -> None:
    """A confusion matrix is a sufficient statistic, which is what makes this safe for
    marginal intervals when a run's per-claim file is lost."""
    gold = [LABELS[i % 4] for i in range(240)]
    pred = [LABELS[(i + 1) % 4] if i % 5 == 0 else gold[i] for i in range(240)]
    original = score(gold, pred)

    pairs = pairs_from_confusion(original.as_dict()["confusion"])
    recovered = score([g for g, _ in pairs], [p for _, p in pairs])

    assert len(pairs) == len(gold)
    assert recovered.macro_f1 == pytest.approx(original.macro_f1)
    assert recovered.accuracy == pytest.approx(original.accuracy)


def test_confusion_expansion_does_not_preserve_claim_order() -> None:
    """The limit of the trick, asserted so nobody pairs two expanded runs together.

    Expanded pairs are grouped by gold label, so position i in one expansion is not the
    same claim as position i in another. Marginal intervals are fine; a paired test on
    expanded data would be meaningless.
    """
    gold = [LABELS[i % 4] for i in range(80)]
    pred = [LABELS[(i + 1) % 4] if i % 5 == 0 else gold[i] for i in range(80)]
    pairs = pairs_from_confusion(score(gold, pred).as_dict()["confusion"])
    assert [g for g, _ in pairs] != gold, "if order survived, pairing would be tempting"


# --- proportions ----------------------------------------------------------------------


def test_a_rate_interval_brackets_the_rate() -> None:
    iv = proportion_interval(43, 100, n_resamples=2000)
    assert iv.point == pytest.approx(0.43)
    assert iv.low < 0.43 < iv.high


def test_attack_rates_on_100_targets_are_about_ten_points_wide_each_side() -> None:
    """The C1 tables are proportions over 100 targets. This is why 0.230 against 0.310
    is a weaker claim than 0.430 against 0.230, and the report should show which."""
    iv = proportion_interval(30, 100, n_resamples=3000)
    assert 0.15 <= iv.width <= 0.25


def test_a_zero_rate_has_a_one_sided_interval() -> None:
    """`poison kept 0.000` must not be reported as exactly zero with no uncertainty."""
    iv = proportion_interval(0, 100, n_resamples=1000)
    assert iv.point == 0.0
    assert iv.low == 0.0 and iv.high == 0.0, (
        "a percentile bootstrap cannot see above an all-zero sample; the report must say "
        "'0 of 100 observed' rather than implying a tight interval around zero"
    )


def test_no_trials_is_refused() -> None:
    with pytest.raises(ValueError, match="no trials"):
        proportion_interval(0, 0)


def test_the_interval_formats_for_a_table() -> None:
    assert str(proportion_interval(43, 100, n_resamples=500)).startswith("0.430 [")
