"""C1's second attack surface: poisoning the evidence corpus (P3.6).

Memory poisoning writes to the experience store. This writes to the pool the retriever
searches -- what an adversary gets by publishing pages somewhere a crawler reaches.

**The two attacks separate the two defences, which is the point of running both.**

* A poisoned *trajectory* asserts a verdict its own evidence does not carry, so C2's
  evaluator gate has something to notice.
* A poisoned *page* is written about the target claim, in detail, with the stance the
  adversary wants. It is genuinely relevant, and the evaluator should score it **high** --
  correctly. Relevance is not truthfulness, and CRAG's grading only measures the former.

So C2 is expected to fail here, and this runner measures that rather than asserting it:
every cell reports the evaluator's mean grade on fabricated strips beside its mean grade
on real ones. If the fakes score as high as the truth, the gate provably cannot separate
them, and the only defence left is knowing who published them -- C3.

**Two adversaries, as in the memory sweep.** `--adversary naive` cites known
misinformation domains, which trust rejects on its prior alone; `--adversary adaptive`
cites the domains the clean history trusts most, which defeats C3 by construction. For
source poisoning the adaptive case is close to worst-case: with C2 unable to help and C3
impersonated, nothing in the pipeline is left to object.

**Ethics (rule 10).** The poisoned corpus is assembled in memory for one run and
discarded. Fabricated pages are never written into the real pool, never served, and
never published. Domains come from AVeriTeC's own misinformation list.

    python -m eval.run_source_poisoning --data data/averitec/dev.json \
        --store data/averitec/pool_dev.jsonl \
        --trajectories data/trajectories/trajectories.jsonl \
        --evaluator results/<date>_t5-evaluator/model --targets 100
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.crag import crag
from attacks.memory import (
    Target,
    choose_targets,
    load_misinfo_domains,
    trusted_domains,
    write_targets,
)
from attacks.sources import poison_pool, poisoned_urls
from core.config import RunConfig
from core.llm import LLMClient
from core.parallel import Progress, map_ordered, resolve_workers
from core.paths import DATA_DIR, announce_metrics, run_dir
from eval.datasets import Claim, load_averitec
from eval.metrics import attack_success
from eval.metrics import score as score_metrics
from eval.run_poisoning import build_trust
from evaluator.score import load_scorer
from experience.format import Trajectory, read_jsonl
from retrieval.store import PooledKnowledgeStore

# Pages per target, not a fraction of the corpus. An adversary publishing pages does not
# get to choose what share of the web they are; they choose how much they write.
PAGES = (0, 1, 3, 5)

# (source_trust,) -- C2's evaluator grading is intrinsic to CRAG and cannot be switched
# off without ceasing to be CRAG, so the comparison here is trust against no trust. That
# C2 is on in *both* rows is exactly why its failure shows up.
CONFIGS: dict[str, bool] = {
    "undefended": False,
    "source_trust": True,
}


@dataclass
class Cell:
    config: str
    pages: int
    attack: dict[str, Any] = field(default_factory=dict)
    macro_f1: float = 0.0
    accuracy: float = 0.0
    grades: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "pages_per_target": self.pages,
            **self.attack,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "grades": self.grades,
            "retrieval": self.retrieval,
        }


def grade_separation(trajectories: Sequence[Trajectory], fake_urls: set[str]) -> dict[str, Any]:
    """Did the evaluator rate fabricated evidence lower than real evidence?

    This is the empirical content of "C2 cannot defend the evidence path". If the two
    means are close, the gate has no signal to act on and saying so is a result, not an
    excuse. Reported per cell because it is a property of the attack, not an assumption.
    """
    fake = [e.grade for t in trajectories for e in t.evidence if e.url in fake_urls]
    real = [e.grade for t in trajectories for e in t.evidence if e.url not in fake_urls]
    out: dict[str, Any] = {
        "n_fake_strips": len(fake),
        "n_real_strips": len(real),
        "mean_grade_fake": round(sum(fake) / len(fake), 4) if fake else None,
        "mean_grade_real": round(sum(real) / len(real), 4) if real else None,
    }
    if fake and real:
        out["separation"] = round(out["mean_grade_real"] - out["mean_grade_fake"], 4)
    return out


def run_cell(
    config: str,
    pages: int,
    claims: Sequence[Claim],
    targets: Sequence[Target],
    rows: Sequence[dict[str, Any]],
    llm: LLMClient,
    scorer: Any,
    trust: Any,
    cfg_base: RunConfig,
    domains: Sequence[str],
    seed: int,
    workers: int = 1,
) -> tuple[Cell, dict[str, str]]:
    use_trust = CONFIGS[config]
    cfg = cfg_base.with_(source_trust=use_trust)

    poisoned_rows, stats = poison_pool(rows, targets, pages, seed, domains)
    fake_urls = poisoned_urls(poisoned_rows)
    # In memory only: the fabricated pages never reach the pool on disk (rule 10).
    source = PooledKnowledgeStore.from_rows(poisoned_rows)

    frozen = frozenset(fake_urls)

    def verify(claim: Claim) -> Trajectory:
        return crag(
            claim,
            llm,
            source,
            scorer,
            cfg,
            trust=trust if use_trust else None,
            fake_urls=frozen,
        )

    progress = Progress(len(claims))
    trajectories = map_ordered(verify, claims, workers, on_done=progress.tick)
    verdicts: dict[str, str] = {
        str(c.id): t.verdict for c, t in zip(claims, trajectories, strict=True)
    }

    labelled = [t for t in trajectories if t.gold_label]
    m = score_metrics([t.gold_label for t in labelled], [t.verdict for t in labelled])

    # Trust's own bookkeeping, summed over claims.
    kept = sum(t.metadata.get("trust", {}).get("poisoned_strips_kept", 0) for t in trajectories)
    seen = sum(
        t.metadata.get("trust", {}).get("poisoned_strips_considered", 0) for t in trajectories
    )
    retrieval = {
        **stats,
        "poisoned_strips_retrieved": sum(
            1 for t in trajectories for e in t.evidence if e.url in fake_urls
        ),
        "poisoned_strips_considered_by_trust": seen,
        "poisoned_strips_kept_by_trust": kept,
        "poison_pass_rate": round(kept / seen, 4) if seen else None,
    }
    return (
        Cell(
            config=config,
            pages=pages,
            macro_f1=m.macro_f1,
            accuracy=m.accuracy,
            grades=grade_separation(trajectories, fake_urls),
            retrieval=retrieval,
        ),
        verdicts,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.run_source_poisoning")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True, help="the clean evidence pool")
    p.add_argument("--trajectories", type=Path, required=True, help="clean history, for trust")
    p.add_argument("--evaluator", type=Path, default=None)
    p.add_argument("--misinfo", type=Path, default=DATA_DIR / "averitec" / "misinfo_list.txt")
    p.add_argument("--targets", type=int, default=100)
    p.add_argument("--pages", type=int, nargs="*", default=list(PAGES))
    p.add_argument("--configs", nargs="*", default=list(CONFIGS))
    p.add_argument("--adversary", choices=["naive", "adaptive"], default="naive")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="run claims concurrently. Almost all the wall-clock is spent blocked on the teacher endpoint, so 8 workers is roughly a 4x speedup. Verdicts are unchanged: every call is cached by input hash and nothing depends on claim order.",
    )
    p.add_argument("--name", default="source-poisoning")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if args.evaluator is None:
        print(
            "[verimem] WARNING: no --evaluator. The grade-separation number is the whole "
            "argument that C2 cannot defend this path, and a lexical stub would not "
            "support it either way."
        )
    if 0 not in args.pages:
        print(
            "[verimem] WARNING: no 0-page cell. Attack success is measured against the "
            "unpoisoned verdicts, so without it there is no baseline to compare to."
        )

    all_claims = load_averitec(args.data, args.data.stem)
    # The same targets as the memory sweep, so the two attacks are comparable.
    targets = choose_targets(all_claims, args.targets, args.seed)
    target_ids = {t.claim_id for t in targets}
    evaluated = [c for c in all_claims if str(c.id) in target_ids]

    rows = [
        json.loads(ln) for ln in args.store.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    clean = read_jsonl(args.trajectories)

    n_cells = len(args.configs) * len(args.pages)
    print(f"[verimem] {len(targets)} targets, {len(evaluated)} claims evaluated per cell")
    print(f"[verimem] clean pool: {len(rows)} pages")
    print(f"[verimem] adversary: {args.adversary}")
    print(f"[verimem] {n_cells} cells x {len(evaluated)} claims = {n_cells * len(evaluated)} runs")
    if args.dry_run:
        print("[verimem] --dry-run: nothing run.")
        return 0

    workers = resolve_workers(args.workers)
    if workers > 1:
        print(f"[verimem] {workers} workers")
    llm = LLMClient()
    scorer = load_scorer(args.evaluator)
    trust = build_trust(clean)

    if args.adversary == "adaptive":
        domains = trusted_domains(clean, trust)
        print(f"[verimem] adaptive adversary impersonates: {', '.join(domains)}")
        print(f"[verimem]   their trust: {[round(trust.trust(d), 3) for d in domains]}")
    else:
        domains = load_misinfo_domains(args.misinfo)
        print(f"[verimem] naive adversary draws from {len(domains)} misinformation domains")

    cfg_base = RunConfig(split=args.data.stem, seed=args.seed)
    cfg_base.api_model, cfg_base.api_base_url = llm.model, llm.base_url

    out = run_dir(args.name)
    cfg_base.save(out)
    write_targets(targets, out / "targets.jsonl")

    cells: list[Cell] = []
    clean_verdicts: dict[str, dict[str, str]] = {}

    for config in args.configs:
        for pages in args.pages:
            print(f"\n[verimem] === {config} @ {pages} pages/target ===", flush=True)
            cell, verdicts = run_cell(
                config,
                pages,
                evaluated,
                targets,
                rows,
                llm,
                scorer,
                trust,
                cfg_base,
                domains,
                args.seed,
                workers,
            )
            if pages == 0:
                clean_verdicts[config] = verdicts
            cell.attack = attack_success(verdicts, targets, clean_verdicts.get(config)).as_dict()
            cells.append(cell)
            g = cell.grades
            print(
                f"    ASR {cell.attack['attack_success_rate']:.3f}  "
                f"macro-F1 {cell.macro_f1:.3f}  "
                f"fake strips retrieved {cell.retrieval['poisoned_strips_retrieved']}  "
                f"grade fake {g.get('mean_grade_fake')} vs real {g.get('mean_grade_real')}"
            )

    payload = {
        "cells": [c.as_dict() for c in cells],
        "n_targets": len(targets),
        "n_evaluated": len(evaluated),
        "adversary": args.adversary,
        "attack_domains": list(domains),
    }
    announce_metrics(out, payload)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n[verimem] === attack success rate by pages per target ===")
    print("  " + "config".ljust(16) + "".join(f"{n:>9}" for n in args.pages))
    for config in args.configs:
        row = {c.pages: c.attack["attack_success_rate"] for c in cells if c.config == config}
        print(
            "  "
            + config.ljust(16)
            + "".join(f"{row.get(n, float('nan')):>9.3f}" for n in args.pages)
        )

    print("\n[verimem] === can the evaluator tell fabricated evidence apart? ===")
    for c in cells:
        if c.grades.get("separation") is not None:
            print(
                f"  {c.config:<16}{c.pages:>3}p  real {c.grades['mean_grade_real']:+.3f}  "
                f"fake {c.grades['mean_grade_fake']:+.3f}  "
                f"separation {c.grades['separation']:+.3f}"
            )
    print(f"\n[verimem] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
