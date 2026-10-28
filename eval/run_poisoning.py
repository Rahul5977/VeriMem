"""C1: the memory-poisoning sweep that tests H1 and H2.

Three configurations, four poison rates, one table:

* **no memory** -- CRAG with experience retrieval off. The adversary has nothing to
  write to, so attack success should stay flat at every rate. This is H1's control, and
  if it moves, something is wrong with the harness rather than the system.
* **memory** -- plain experience retrieval, ExpRAG as published. **H1** predicts this is
  where attack success climbs with the poison rate.
* **gate only** / **trust only** -- each defence alone, so the combined result can be
  attributed. H2 is not answered by showing the pair works; it needs to say which one does.
* **gated memory** -- the same, plus C2's evaluator gate and C3's source trust.
  **H2** predicts this cuts attack success relative to plain memory.

**Two adversaries, because the naive one flatters the defence.** `--adversary naive`
cites known misinformation domains, which the system has never seen -- and C3 rejects an
unseen domain on its prior alone, without judging anything it says. `--adversary adaptive`
cites the domains the clean history trusts most, so C3 is defeated by construction and
only C2's gate is left. The second number is the one that says whether the defence
understands poisoning or merely distrusts newcomers.

**The store is rebuilt per rate, and the targets never change.** Poisoning at 10% must
differ from 5% only in how much fake experience exists -- same seed, same targets, same
real trajectories underneath -- or the comparison measures sampling noise.

**What is reported.** Attack success is the headline, but on its own it cannot tell a
surgical attack from a destructive one, so each cell also carries macro-F1 over the full
evaluation set and the share of poisoned records that survived retrieval. A defence that
cuts attack success by dropping *all* experience has not defended anything.

    python -m eval.run_poisoning --data data/averitec/dev.json \
        --store data/averitec/pool_dev.jsonl \
        --trajectories data/trajectories/trajectories.jsonl \
        --evaluator results/<date>_t5-evaluator/model --targets 100
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.crag import crag
from attacks.memory import (
    Target,
    choose_targets,
    load_misinfo_domains,
    poison_store,
    trusted_domains,
    write_targets,
)
from core.config import RunConfig
from core.llm import LLMClient
from core.paths import DATA_DIR, announce_metrics, run_dir
from eval.datasets import Claim, load_averitec, sample
from eval.metrics import attack_success
from eval.metrics import score as score_metrics
from evaluator.score import load_scorer
from experience.format import Trajectory, read_jsonl
from experience.store import BetaTrust, ExperienceStore
from retrieval.store import open_store

RATES = (0.0, 0.05, 0.10, 0.20)

# flag triples: (experience_retrieval, gate_experience, source_trust)
#
# `gate_only` and `trust_only` exist because the combined defence is not interpretable on
# its own. In the first full sweep, gated_memory dropped every poisoned record -- but the
# naive adversary cites domains the system has never seen, so C3 could reject them as
# strangers without ever judging their content. Running each defence alone says which one
# is load-bearing, and against which adversary.
CONFIGS: dict[str, tuple[bool, bool, bool]] = {
    "no_memory": (False, False, False),
    "memory": (True, False, False),
    "gate_only": (True, True, False),
    "trust_only": (True, False, True),
    "gated_memory": (True, True, True),
}


@dataclass
class Cell:
    """One (configuration, rate) square of the table."""

    config: str
    rate: float
    attack: dict[str, Any]
    macro_f1: float
    accuracy: float
    retrieval: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "rate": self.rate,
            **self.attack,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "retrieval": self.retrieval,
        }


def build_trust(trajectories: Sequence[Trajectory]) -> BetaTrust:
    """C3 trust learned from the clean history, before any poisoning.

    Trust must be fitted on what the system legitimately saw. Letting it observe the
    poisoned store would hand the defence the answer -- it would learn the attack
    domains are bad *from the attack itself*, which no deployed system could do before
    being attacked.
    """
    trust = BetaTrust()
    for t in trajectories:
        trust.observe(t)
    return trust


def run_cell(
    config: str,
    rate: float,
    claims: Sequence[Claim],
    targets: Sequence[Target],
    clean: Sequence[Trajectory],
    llm: LLMClient,
    source,
    scorer,
    trust: BetaTrust,
    cfg_base: RunConfig,
    domains: Sequence[str],
    seed: int,
) -> tuple[Cell, dict[str, str]]:
    retrieval_on, gate, use_trust = CONFIGS[config]
    cfg = cfg_base.with_(
        experience_retrieval=retrieval_on, gate_experience=gate, source_trust=use_trust
    )

    memory = None
    poison_stats: dict[str, Any] = {}
    if retrieval_on:
        store_trajectories, poison_stats = poison_store(clean, targets, rate, seed, domains)
        memory = ExperienceStore(
            store_trajectories,
            scorer=scorer,
            trust=trust,
            gate_threshold=cfg.gate_threshold,
            trust_floor=cfg.trust_floor,
        )

    verdicts: dict[str, str] = {}
    trajectories = []
    for i, claim in enumerate(claims, start=1):
        t = crag(claim, llm, source, scorer, cfg, memory=memory)
        trajectories.append(t)
        verdicts[str(claim.id)] = t.verdict
        if i % 25 == 0 or i == len(claims):
            print(f"    [{i}/{len(claims)}]", flush=True)

    labelled = [t for t in trajectories if t.gold_label]
    m = score_metrics([t.gold_label for t in labelled], [t.verdict for t in labelled])
    return (
        Cell(
            config=config,
            rate=rate,
            attack={},  # filled by the caller, which holds the clean verdicts
            macro_f1=m.macro_f1,
            accuracy=m.accuracy,
            retrieval={**(memory.stats.as_dict() if memory else {}), **poison_stats},
        ),
        verdicts,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.run_poisoning")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True, help="evidence pool")
    p.add_argument("--trajectories", type=Path, required=True, help="the clean experience store")
    p.add_argument("--evaluator", type=Path, default=None)
    p.add_argument("--misinfo", type=Path, default=DATA_DIR / "averitec" / "misinfo_list.txt")
    p.add_argument("--targets", type=int, default=100)
    p.add_argument("--limit", type=int, default=None, help="claims evaluated per cell")
    p.add_argument("--rates", type=float, nargs="*", default=list(RATES))
    p.add_argument("--configs", nargs="*", default=list(CONFIGS))
    p.add_argument("--seed", type=int, default=13)
    p.add_argument(
        "--adversary",
        choices=["naive", "adaptive"],
        default="naive",
        help="naive cites known misinformation domains; adaptive cites the domains the "
        "clean history trusts most, which defeats C3 by construction and leaves C2 alone",
    )
    p.add_argument("--name", default="poisoning")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if args.evaluator is None:
        print(
            "[verimem] WARNING: no --evaluator. Gating would run on a lexical stub, so "
            "H2 would be testing word counting rather than C2."
        )

    all_claims = load_averitec(args.data, args.data.stem)
    targets = choose_targets(all_claims, args.targets, args.seed)
    # Every target must be evaluated, or its attack cannot be observed.
    target_ids = {t.claim_id for t in targets}
    evaluated = [c for c in all_claims if str(c.id) in target_ids]
    if args.limit:
        extra = [
            c for c in sample(all_claims, args.limit, args.seed) if str(c.id) not in target_ids
        ]
        evaluated = evaluated + extra[: max(0, args.limit - len(evaluated))]

    clean = read_jsonl(args.trajectories)
    domains = load_misinfo_domains(args.misinfo)

    n_cells = sum(len(args.rates) if CONFIGS[c][0] else 1 for c in args.configs)
    print(f"[verimem] {len(targets)} targets, {len(evaluated)} claims evaluated per cell")
    print(f"[verimem] clean store: {len(clean)} trajectories")
    print(f"[verimem] adversary: {args.adversary}")
    print(f"[verimem] {n_cells} cells x {len(evaluated)} claims = {n_cells * len(evaluated)} runs")
    if args.dry_run:
        print("[verimem] --dry-run: nothing run.")
        return 0

    llm = LLMClient()
    scorer = load_scorer(args.evaluator)
    source = open_store(args.store)
    trust = build_trust(clean)

    if args.adversary == "adaptive":
        domains = trusted_domains(clean, trust)
        print(f"[verimem] adaptive adversary impersonates: {', '.join(domains)}")
        print(f"[verimem]   their trust: {[round(trust.trust(d), 3) for d in domains]}")
    else:
        print(f"[verimem] naive adversary draws from {len(domains)} misinformation domains")

    cfg_base = RunConfig(split=args.data.stem, seed=args.seed)
    cfg_base.api_model, cfg_base.api_base_url = llm.model, llm.base_url

    out = run_dir(args.name)
    cfg_base.save(out)
    write_targets(targets, out / "targets.jsonl")

    cells: list[Cell] = []
    clean_verdicts: dict[str, dict[str, str]] = {}

    for config in args.configs:
        rates = args.rates if CONFIGS[config][0] else [0.0]
        for rate in rates:
            print(f"\n[verimem] === {config} @ {rate:.0%} ===", flush=True)
            cell, verdicts = run_cell(
                config,
                rate,
                evaluated,
                targets,
                clean,
                llm,
                source,
                scorer,
                trust,
                cfg_base,
                domains,
                args.seed,
            )
            if rate == 0.0:
                clean_verdicts[config] = verdicts
            cell.attack = attack_success(verdicts, targets, clean_verdicts.get(config)).as_dict()
            cell.attack["clean_macro_f1"] = cells[0].macro_f1 if cells else cell.macro_f1
            cells.append(cell)
            print(
                f"    ASR {cell.attack['attack_success_rate']:.3f}  "
                f"macro-F1 {cell.macro_f1:.3f}  "
                f"poison kept {cell.retrieval.get('poison_pass_rate')}"
            )

    table = [c.as_dict() for c in cells]
    payload = {
        "cells": table,
        "n_targets": len(targets),
        "n_evaluated": len(evaluated),
        "adversary": args.adversary,
        "attack_domains": list(domains),
    }
    announce_metrics(out, payload)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "cells": table,
                "n_targets": len(targets),
                "n_evaluated": len(evaluated),
                "adversary": args.adversary,
                "attack_domains": list(domains),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[verimem] === attack success rate ===")
    header = "  " + "config".ljust(14) + "".join(f"{r:>9.0%}" for r in args.rates)
    print(header)
    for config in args.configs:
        row = [c for c in cells if c.config == config]
        cellmap = {c.rate: c.attack["attack_success_rate"] for c in row}
        line = "  " + config.ljust(14)
        for r in args.rates:
            v = cellmap.get(r, cellmap.get(0.0))
            line += f"{v:>9.3f}"
        print(line)
    print(f"\n[verimem] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
