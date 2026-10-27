"""Accuracy and macro-F1 over the four verdicts (P0.10).

Written by hand rather than pulled from scikit-learn: it is twenty lines, it removes a
dependency from the one module every experiment imports, and macro-F1 has a judgement
call in it -- what to do with a class that never appears -- that should be explicit
rather than inherited. A class with no gold instances and no predictions is skipped; a
class that is predicted but never correct scores zero and is counted. That matches
`sklearn.metrics.f1_score(average="macro", zero_division=0)` over the labels present.
"""

from __future__ import annotations

from collections.abc import Sequence
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
