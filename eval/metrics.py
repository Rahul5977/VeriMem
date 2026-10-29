"""Accuracy and macro-F1 over the four verdicts (P0.10).

Written by hand rather than pulled from scikit-learn: it is twenty lines, it removes a
dependency from the one module every experiment imports, and macro-F1 has a judgement
call in it -- what to do with a class that never appears -- that should be explicit
rather than inherited. A class with no gold instances and no predictions is skipped; a
class that is predicted but never correct scores zero and is counted. That matches
`sklearn.metrics.f1_score(average="macro", zero_division=0)` over the labels present.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from experience.format import VERDICTS


@dataclass
class ClassScore:
    label: str
    precision: float
    recall: float
    f1: float
    support: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.support,
        }


@dataclass
class Metrics:
    accuracy: float = 0.0
    macro_f1: float = 0.0
    n: int = 0
    per_class: dict[str, ClassScore] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    unparseable: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "unparseable": self.unparseable,
            "per_class": {k: v.as_dict() for k, v in self.per_class.items()},
            "confusion": self.confusion,
        }

    def summary(self) -> str:
        head = f"n={self.n}  accuracy={self.accuracy:.3f}  macro-F1={self.macro_f1:.3f}"
        rows = "\n".join(
            f"  {label:<22} P {s.precision:.3f}  R {s.recall:.3f}  F1 {s.f1:.3f}  n={s.support}"
            for label, s in self.per_class.items()
        )
        return f"{head}\n{rows}" if rows else head


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def score(
    gold: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str] = VERDICTS,
    unparseable: int = 0,
) -> Metrics:
    """Accuracy, macro-F1, per-class P/R/F1 and a confusion matrix."""
    if len(gold) != len(predicted):
        raise ValueError(f"gold has {len(gold)} items, predicted has {len(predicted)}")
    if not gold:
        return Metrics(unparseable=unparseable)

    correct = sum(g == p for g, p in zip(gold, predicted, strict=True))
    metrics = Metrics(
        accuracy=correct / len(gold),
        n=len(gold),
        unparseable=unparseable,
        confusion={g: dict.fromkeys(labels, 0) for g in labels},
    )

    for g, p in zip(gold, predicted, strict=True):
        if g in metrics.confusion and p in metrics.confusion[g]:
            metrics.confusion[g][p] += 1

    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
        if tp + fn == 0 and tp + fp == 0:
            continue  # class absent from both gold and predictions: not scored
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = _f1(precision, recall)
        metrics.per_class[label] = ClassScore(label, precision, recall, f1, tp + fn)
        f1s.append(f1)

    metrics.macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return metrics


def majority_baseline(gold: Sequence[str]) -> Metrics:
    """Score of always predicting the most common gold label.

    AVeriTeC is heavily skewed -- "Refuted" is 61% of dev and 57% of train -- so an
    accuracy figure means little on its own. Every results table should carry this row,
    because a system below it has learned nothing.
    """
    if not gold:
        return Metrics()
    most_common = max(set(gold), key=list(gold).count)
    return score(gold, [most_common] * len(gold))


@dataclass
class AttackResult:
    """How well a poisoning attack worked, and how much it cost the system overall."""

    n_targets: int = 0
    n_succeeded: int = 0
    n_degraded: int = 0
    clean_macro_f1: float | None = None
    poisoned_macro_f1: float | None = None

    @property
    def attack_success_rate(self) -> float:
        """Fraction of targets where the system returned the adversary's chosen verdict."""
        return self.n_succeeded / self.n_targets if self.n_targets else 0.0

    @property
    def collateral(self) -> float | None:
        """Macro-F1 lost across all claims, not just the targets.

        A quiet attack that flips its targets and leaves everything else intact is more
        dangerous than a loud one that wrecks the system, because nobody notices it.
        Reporting only attack success would make the quiet attack look identical to the
        loud one.
        """
        if self.clean_macro_f1 is None or self.poisoned_macro_f1 is None:
            return None
        return round(self.clean_macro_f1 - self.poisoned_macro_f1, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_targets": self.n_targets,
            "n_succeeded": self.n_succeeded,
            "attack_success_rate": round(self.attack_success_rate, 4),
            "n_degraded": self.n_degraded,
            "clean_macro_f1": self.clean_macro_f1,
            "poisoned_macro_f1": self.poisoned_macro_f1,
            "collateral_macro_f1": self.collateral,
        }


def attack_success(
    verdicts: dict[str, str],
    targets: Sequence[Any],
    clean_verdicts: dict[str, str] | None = None,
) -> AttackResult:
    """Score a poisoning run.

    `verdicts` and `clean_verdicts` map claim id -> verdict, under attack and without.
    A target *succeeds* when the system returns the adversary's chosen verdict. It is
    *degraded* when the attack merely broke a previously correct answer without
    achieving the intended one -- harm, but not control, and worth separating because
    H2 predicts defences should cut success faster than they cut degradation.
    """
    result = AttackResult(n_targets=len(targets))
    for t in targets:
        got = verdicts.get(str(t.claim_id))
        if got is None:
            continue
        if got == t.target_verdict:
            result.n_succeeded += 1
        elif clean_verdicts is not None:
            was = clean_verdicts.get(str(t.claim_id))
            if was == t.gold_label and got != t.gold_label:
                result.n_degraded += 1
    return result


# --- uncertainty ----------------------------------------------------------------------
#
# Macro-F1 on AVeriTeC dev weights four classes equally while two of them have ~36
# instances each, and every system scores near zero on those. So a quarter of the metric
# rests on one or two correct predictions, and a handful of claims changing moves the
# headline by 0.05. That is how CRAG read as 0.414 in one run and 0.363 in another with
# no change to the pipeline that could explain it.
#
# The fix is not more runs -- three seeds costs three times the compute for three points.
# Resampling the claims of a single run gives the sampling distribution directly, needs
# no model calls at all, and answers the question actually being asked: would this
# difference survive a different draw of 500 claims from the same population?


@dataclass
class Interval:
    """A point estimate with a percentile bootstrap interval."""

    point: float
    low: float
    high: float
    level: float = 0.95

    @property
    def width(self) -> float:
        return self.high - self.low

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"

    def as_dict(self) -> dict[str, Any]:
        return {
            "point": round(self.point, 4),
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "level": self.level,
        }


def pairs_from_confusion(confusion: Mapping[str, Mapping[str, int]]) -> list[tuple[str, str]]:
    """Expand a confusion matrix back into (gold, predicted) pairs.

    The matrix is a sufficient statistic for every metric here, and bootstrapping
    resamples claims without regard to order, so the expansion loses nothing that
    matters. It exists because a run's per-claim file does not always survive -- the
    G1 re-run came back only as the metrics its kernel printed to the log -- and a
    confidence interval should not require re-running the experiment.
    """
    out: list[tuple[str, str]] = []
    for gold, row in confusion.items():
        for pred, count in row.items():
            out.extend([(gold, pred)] * int(count))
    return out


def bootstrap_metric(
    gold: Sequence[str],
    pred: Sequence[str],
    metric: str = "macro_f1",
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 13,
) -> Interval:
    """Percentile bootstrap over claims.

    Resamples (gold, predicted) pairs with replacement, which treats the evaluation set
    as a sample from the claim population -- the assumption a reader makes when reading
    one number as an estimate of a system's quality.
    """
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold labels but {len(pred)} predictions")
    if not gold:
        raise ValueError("nothing to bootstrap")

    rng = random.Random(seed)
    n = len(gold)
    point = getattr(score(gold, pred), metric)

    draws: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(getattr(score([gold[i] for i in idx], [pred[i] for i in idx]), metric))
    draws.sort()

    tail = (1.0 - level) / 2.0
    lo = draws[int(tail * n_resamples)]
    hi = draws[min(int((1.0 - tail) * n_resamples), n_resamples - 1)]
    return Interval(point=point, low=lo, high=hi, level=level)


def bootstrap_difference(
    gold: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    metric: str = "macro_f1",
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 13,
) -> Interval:
    """The interval for A minus B, resampling both systems on the **same** claims.

    This is the comparison that matters and it is not the same as checking whether two
    separate intervals overlap. Two systems evaluated on one claim set are correlated --
    they fail on the same hard claims -- so the difference is estimated far more precisely
    than either absolute score. Overlapping intervals routinely hide a difference that is
    reliably non-zero, which would read as "no effect" when there is one.

    An interval excluding zero means the sign of the difference is stable across
    resamples of the claim set.
    """
    if not (len(gold) == len(pred_a) == len(pred_b)):
        raise ValueError("the two systems must be scored on the same claims, in the same order")
    if not gold:
        raise ValueError("nothing to bootstrap")

    rng = random.Random(seed)
    n = len(gold)
    point = getattr(score(gold, pred_a), metric) - getattr(score(gold, pred_b), metric)

    draws: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        a = getattr(score(g, [pred_a[i] for i in idx]), metric)
        b = getattr(score(g, [pred_b[i] for i in idx]), metric)
        draws.append(a - b)
    draws.sort()

    tail = (1.0 - level) / 2.0
    lo = draws[int(tail * n_resamples)]
    hi = draws[min(int((1.0 - tail) * n_resamples), n_resamples - 1)]
    return Interval(point=point, low=lo, high=hi, level=level)


def proportion_interval(
    successes: int, trials: int, level: float = 0.95, seed: int = 13, n_resamples: int = 10_000
) -> Interval:
    """An interval for a rate like attack success.

    C1's headline numbers are proportions over 100 targets, where the standard error is
    near 0.05 -- so 0.230 against 0.310 is a far weaker claim than 0.430 against 0.230,
    and the table should let a reader see which is which.
    """
    if trials <= 0:
        raise ValueError("no trials")
    rng = random.Random(seed)
    point = successes / trials
    outcomes = [1] * successes + [0] * (trials - successes)
    draws = sorted(
        sum(outcomes[rng.randrange(trials)] for _ in range(trials)) / trials
        for _ in range(n_resamples)
    )
    tail = (1.0 - level) / 2.0
    return Interval(
        point=point,
        low=draws[int(tail * n_resamples)],
        high=draws[min(int((1.0 - tail) * n_resamples), n_resamples - 1)],
        level=level,
    )
