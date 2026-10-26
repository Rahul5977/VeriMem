"""Tune CRAG's two thresholds on dev, and report the action shares (P1.7).

The defaults (+-0.6) were guessed before the evaluator existed. The trained model puts
relevant passages at a mean of +0.28 and irrelevant ones at -0.54, so +-0.6 would route
almost everything to "incorrect" and the CRAG row would lose to plain RAG for reasons
that have nothing to do with CRAG.

The sweep would be ruinous done naively -- a grid of 100 threshold pairs times 500
claims is 50,000 pipeline runs. But a claim's *grade* does not depend on the thresholds,
and neither does what each branch would answer. So each claim is run once per branch
(three verdicts), and after that every grid point is a lookup. The expensive part is
paid once and cached; the sweep itself takes milliseconds.

    python -m evaluator.thresholds --data data/averitec/dev.json \
        --store data/averitec/pool_dev.jsonl --limit 150
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.crag import run_crag
from core.config import RunConfig
from core.llm import LLMClient
from core.paths import run_dir
from eval.datasets import load_averitec, sample
from eval.metrics import score as score_metrics
from evaluator.score import load_scorer
from retrieval.store import open_store

BRANCHES = ("correct", "incorrect", "ambiguous")


@dataclass
class Outcome:
    """What each branch answered for one claim, plus the grade that picks between them."""

    claim_id: str
    gold_label: str | None
    grade: float
    verdicts: dict[str, str]

    def verdict_for(self, upper: float, lower: float) -> str:
        if self.grade >= upper:
            return self.verdicts["correct"]
        if self.grade <= lower:
            return self.verdicts["incorrect"]
        return self.verdicts["ambiguous"]

    def action_for(self, upper: float, lower: float) -> str:
        if self.grade >= upper:
            return "correct"
        return "incorrect" if self.grade <= lower else "ambiguous"


def precompute(claims, llm, source, scorer, cfg: RunConfig) -> list[Outcome]:
    """Run all three branches per claim. This is the only expensive part."""
    out = []
    for i, claim in enumerate(claims, start=1):
        verdicts, grade = {}, -1.0
        for branch in BRANCHES:
            result = run_crag(claim, llm, source, scorer, cfg, force_action=branch)
            verdicts[branch] = result.verdict
            grade = result.grade
        out.append(Outcome(str(claim.id), claim.gold_label, grade, verdicts))
        if i % 10 == 0 or i == len(claims):
            print(f"  [{i}/{len(claims)}]", flush=True)
    return out


def sweep(
    outcomes: Sequence[Outcome],
    uppers: Sequence[float],
    lowers: Sequence[float],
) -> list[dict[str, Any]]:
    """Macro-F1 at every (upper, lower) pair. Lower must stay below upper."""
    labelled = [o for o in outcomes if o.gold_label]
    gold = [o.gold_label for o in labelled]
    rows = []
    for upper in uppers:
        for lower in lowers:
            if lower >= upper:
                continue
            predicted = [o.verdict_for(upper, lower) for o in labelled]
            m = score_metrics(gold, predicted)
            actions = [o.action_for(upper, lower) for o in labelled]
            rows.append(
                {
                    "upper": round(upper, 3),
                    "lower": round(lower, 3),
                    "macro_f1": m.macro_f1,
                    "accuracy": m.accuracy,
                    "action_share": {
                        b: round(actions.count(b) / len(actions), 3) for b in BRANCHES
                    },
                }
            )
    return sorted(rows, key=lambda r: -r["macro_f1"])


def branch_only(outcomes: Sequence[Outcome], branch: str) -> dict[str, float]:
    """What always taking one branch would score -- the floor routing has to beat."""
    labelled = [o for o in outcomes if o.gold_label]
    m = score_metrics([o.gold_label for o in labelled], [o.verdicts[branch] for o in labelled])
    return {"macro_f1": m.macro_f1, "accuracy": m.accuracy}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evaluator.thresholds")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument(
        "--evaluator",
        type=Path,
        default=None,
        help="trained model dir; omitted means the lexical stub",
    )
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--steps", type=int, default=13, help="grid points per threshold")
    p.add_argument("--name", default="thresholds")
    args = p.parse_args(argv)

    claims = sample(load_averitec(args.data, "dev"), args.limit, args.seed)
    cfg = RunConfig(split="dev", limit=args.limit, seed=args.seed)
    llm = LLMClient()
    scorer = load_scorer(args.evaluator)
    source = open_store(args.store)

    print(
        f"[verimem] {len(claims)} claims x {len(BRANCHES)} branches = "
        f"{len(claims) * len(BRANCHES)} pipeline runs (cached ones are instant)"
    )
    if args.evaluator is None:
        print(
            "[verimem] WARNING: no --evaluator, using the lexical stub. Thresholds tuned "
            "against the stub do not transfer to the trained model."
        )

    outcomes = precompute(claims, llm, source, scorer, cfg)

    grid = [-0.9 + i * 1.8 / (args.steps - 1) for i in range(args.steps)]
    rows = sweep(outcomes, grid, grid)
    if not rows:
        raise SystemExit("Empty grid: no (upper, lower) pair with lower < upper.")

    best = rows[0]
    out = run_dir(args.name)
    (out / "thresholds.json").write_text(
        json.dumps(
            {
                "best": best,
                "always_one_branch": {b: branch_only(outcomes, b) for b in BRANCHES},
                "top_10": rows[:10],
                "grades": [round(o.grade, 4) for o in outcomes],
                "outcomes": [asdict(o) for o in outcomes],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\n[verimem] best: upper={best['upper']} lower={best['lower']}  "
        f"macro-F1 {best['macro_f1']:.3f}  accuracy {best['accuracy']:.3f}"
    )
    print(f"[verimem] action share at best: {best['action_share']}")
    print("\n[verimem] always one branch (the floor routing must beat):")
    for b in BRANCHES:
        s = branch_only(outcomes, b)
        print(f"    {b:<10} macro-F1 {s['macro_f1']:.3f}  accuracy {s['accuracy']:.3f}")
    print(f"\n[verimem] wrote {out}/thresholds.json")
    print(
        f"[verimem] put these in RunConfig: crag_upper_threshold={best['upper']}, "
        f"crag_lower_threshold={best['lower']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
