"""Turn results folders into report-ready tables with confidence intervals (P4.1).

Every number in the paper should arrive with the uncertainty that produced it, because
on this benchmark the uncertainty is the story. Macro-F1 weights four classes equally
while two of them hold ~36 of 500 claims, and all systems score near zero on those -- so
a quarter of the metric rests on one or two correct predictions. CRAG read as 0.414 in
one run and 0.363 in another; the paired bootstrap puts the difference against plain RAG
at +0.009 [-0.032, 0.052]. The ordering was never established, and a table of bare point
estimates is what hid that.

Two kinds of interval, and the difference matters:

* **marginal** -- for a single system's score. Computed from a confusion matrix when the
  per-claim file is gone, which is sufficient for this.
* **paired** -- for A minus B on the same claims. Far tighter, because the systems fail on
  the same hard claims. Needs the per-claim trajectories; it cannot be recovered from a
  confusion matrix, and doing so anyway would compare claims that are not the same claim.

    python -m eval.report_tables --run results/2026-10-28_g1-rerun
    python -m eval.report_tables --attack results/2026-10-28_poison-ablation-naive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import (
    Interval,
    bootstrap_difference,
    bootstrap_metric,
    pairs_from_confusion,
    proportion_interval,
)
from experience.format import read_jsonl

SYSTEM_ORDER = ("no_retrieval", "plain_rag", "crag")


def _aligned(run: Path) -> tuple[list[str], dict[str, list[str]]] | None:
    """Per-claim verdicts for every system in a run, aligned by claim id.

    Returns None when the trajectory files are absent, so the caller can fall back to
    marginal intervals instead of silently reporting a paired test it cannot support.
    """
    runs: dict[str, dict[str, object]] = {}
    for path in sorted(run.glob("*_trajectories.jsonl")):
        system = path.name.replace("_trajectories.jsonl", "")
        runs[system] = {t.claim_id: t for t in read_jsonl(path)}
    if len(runs) < 1:
        return None

    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    first = next(iter(runs.values()))
    ids = [i for i in ids if getattr(first[i], "gold_label", None)]
    if not ids:
        return None
    gold = [first[i].gold_label for i in ids]  # type: ignore[union-attr]
    pred = {s: [runs[s][i].verdict for i in ids] for s in runs}  # type: ignore[union-attr]
    return gold, pred


def system_table(run: Path, metric: str = "macro_f1", resamples: int = 10_000) -> None:
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    aligned = _aligned(run)

    print(f"\n### {run.name} — {metric}, 95% bootstrap\n")
    print(f"| system | {metric} | 95% CI |")
    print("|---|---|---|")

    systems = [s for s in SYSTEM_ORDER if s in metrics] or [
        k for k, v in metrics.items() if isinstance(v, dict) and "macro_f1" in v
    ]
    for s in systems:
        if aligned:
            gold, pred = aligned
            iv = bootstrap_metric(gold, pred[s], metric=metric, n_resamples=resamples)
        else:
            pairs = pairs_from_confusion(metrics[s]["confusion"])
            iv = bootstrap_metric(
                [g for g, _ in pairs], [p for _, p in pairs], metric=metric, n_resamples=resamples
            )
        print(f"| {s} | {iv.point:.3f} | [{iv.low:.3f}, {iv.high:.3f}] |")

    if not aligned:
        print(
            "\n*Marginal intervals from the confusion matrix; the per-claim files were "
            "not available, so no paired comparison is reported.*"
        )
        return

    gold, pred = aligned
    print(f"\n**Paired differences** (same {len(gold)} claims)\n")
    print("| comparison | difference | 95% CI | significant |")
    print("|---|---|---|---|")
    for a, b in zip(systems[1:], systems[:-1], strict=True):
        d = bootstrap_difference(gold, pred[a], pred[b], metric=metric, n_resamples=resamples)
        sig = "yes" if (d.low > 0 or d.high < 0) else "**no**"
        print(f"| {a} − {b} | {d.point:+.3f} | [{d.low:+.3f}, {d.high:+.3f}] | {sig} |")
    if len(systems) >= 3:
        d = bootstrap_difference(
            gold, pred[systems[-1]], pred[systems[0]], metric=metric, n_resamples=resamples
        )
        sig = "yes" if (d.low > 0 or d.high < 0) else "**no**"
        print(
            f"| {systems[-1]} − {systems[0]} | {d.point:+.3f} | [{d.low:+.3f}, {d.high:+.3f}] | {sig} |"
        )


def attack_table(run: Path, resamples: int = 10_000) -> None:
    """Attack success rates as proportions over the target set.

    These are marginal intervals. The cells share their targets, so the true precision of
    a *difference* between two cells is better than comparing two intervals suggests --
    but the per-target outcomes are not stored, so a paired test is not available for
    runs already completed. Non-overlapping intervals are therefore sufficient evidence;
    overlapping ones are inconclusive rather than negative.
    """
    d = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    n = d["n_targets"]
    rate_key = "rate" if "rate" in d["cells"][0] else "pages_per_target"

    print(f"\n### {run.name} — attack success rate, 95% bootstrap (n={n})")
    print(f"\nadversary: **{d.get('adversary', 'naive')}**")
    if d.get("attack_domains"):
        print(f"\ndomains: `{', '.join(d['attack_domains'][:5])}`")
    print(f"\n| configuration | {rate_key} | ASR | 95% CI | macro-F1 |")
    print("|---|---|---|---|---|")
    for c in d["cells"]:
        iv: Interval = proportion_interval(
            round(c["attack_success_rate"] * n), n, n_resamples=resamples
        )
        print(
            f"| {c['config']} | {c[rate_key]} | {iv.point:.3f} "
            f"| [{iv.low:.3f}, {iv.high:.3f}] | {c['macro_f1']:.3f} |"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.report_tables")
    p.add_argument("--run", type=Path, action="append", default=[], help="a baselines run")
    p.add_argument("--attack", type=Path, action="append", default=[], help="a poisoning run")
    p.add_argument("--metric", default="macro_f1")
    p.add_argument("--resamples", type=int, default=10_000)
    args = p.parse_args(argv)

    if not args.run and not args.attack:
        p.error("give at least one --run or --attack")
    for run in args.run:
        system_table(run, args.metric, args.resamples)
    for run in args.attack:
        attack_table(run, args.resamples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
