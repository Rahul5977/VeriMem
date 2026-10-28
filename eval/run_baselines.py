"""Run the Phase 0 baselines on AVeriTeC and write a results folder (P0.11, rule 7).

    # the G0 run: 20 claims, both baselines
    python -m eval.run_baselines --data data/averitec/dev.json \
        --store data/averitec/knowledge_store/dev --limit 20 --system plain_rag

    # what it will cost and how many claims have evidence, without calling anything
    python -m eval.run_baselines --data data/averitec/dev.json --limit 20 --dry-run

Rule 2 is enforced here: the estimated cost is printed and, on a paid endpoint,
confirmed before any call is made. On a local endpoint the estimate is zero and the run
proceeds. Every call goes through `core.llm`, so a second run of the same claims is
served from cache and costs nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.baselines import SYSTEMS as BASELINE_SYSTEMS
from agent.crag import crag
from core.config import RunConfig
from core.llm import LLMClient, is_free
from core.paths import announce_metrics, run_dir
from eval.datasets import label_counts, load_averitec, sample
from eval.metrics import score
from evaluator.score import load_scorer
from experience.format import read_jsonl, write_jsonl
from experience.store import BetaTrust, ExperienceStore
from retrieval.store import GoldEvidenceSource, open_store
from training.verifier import load_verifier

# Measured from the prompts in agent/baselines.py; used only for the pre-run estimate.
SYSTEMS = (*BASELINE_SYSTEMS, "crag")
AVG_INPUT_TOKENS = {"no_retrieval": 120, "plain_rag": 1400, "crag": 1200}
AVG_OUTPUT_TOKENS = 90
# Measured on an 8 GB Mac running qwen2.5:7b through Ollama, Oct 2026.
SECONDS_PER_CALL = 18


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m eval.run_baselines")
    p.add_argument("--data", type=Path, required=True, help="AVeriTeC split (.json or .jsonl)")
    p.add_argument("--store", type=Path, default=None, help="knowledge store dir or jsonl")
    p.add_argument(
        "--system",
        choices=[*SYSTEMS, "both", "all"],
        default="both",
        help="'both' is the two baselines; 'all' adds crag",
    )
    p.add_argument(
        "--evaluator",
        type=Path,
        default=None,
        help="trained evaluator dir; required for --system crag/all",
    )
    p.add_argument(
        "--evidence",
        choices=["store", "gold"],
        default="store",
        help="'gold' is an ORACLE upper bound, not a baseline",
    )
    p.add_argument("--split", default="dev")
    p.add_argument(
        "--trajectories",
        type=Path,
        default=None,
        help="past trajectories to retrieve as experience (ExpRAG). Required by "
        "--experience-retrieval.",
    )
    p.add_argument(
        "--experience-retrieval",
        action="store_true",
        help="put k retrieved past verifications in the verifier's context",
    )
    p.add_argument("--gate-experience", action="store_true", help="C2: evaluator-gate them")
    p.add_argument("--source-trust", action="store_true", help="C3: drop low-trust sources")
    p.add_argument(
        "--verifier-base",
        action="store_true",
        help="serve the student model with NO adapter -- the prompted row of the G2 "
        "table. Travels the same path as an adapter so the comparison is weights-only.",
    )
    p.add_argument(
        "--verifier-adapter",
        type=Path,
        default=None,
        help="a LoRA adapter directory from training.sft. The final verdict comes from "
        "the fine-tuned student (C4); the teacher still rewrites and decomposes.",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="claims to run; default is the whole split"
    )
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--name", default=None, help="run name; default <system>-<split>")
    p.add_argument("--dry-run", action="store_true", help="load, estimate, stop")
    p.add_argument("--yes", action="store_true", help="skip the paid-run confirmation")
    return p.parse_args(argv)


def run_one(
    system: str, claims, llm, source, args, scorer=None, cfg=None, verifier=None, memory=None
) -> tuple[list, dict]:
    trajectories = []
    for i, claim in enumerate(claims, start=1):
        try:
            if system == "crag":
                trajectories.append(
                    crag(claim, llm, source, scorer, cfg, memory=memory, verifier=verifier)
                )
            elif system == "plain_rag":
                trajectories.append(BASELINE_SYSTEMS[system](claim, llm, source, k=args.k))
            else:
                trajectories.append(BASELINE_SYSTEMS[system](claim, llm))
        except Exception as exc:  # noqa: BLE001 - any endpoint failure lands here
            if "Connection" not in type(exc).__name__ and "Connection" not in str(exc):
                raise
            # Every completed call is already cached, so this is a pause, not a loss.
            raise SystemExit(
                f"\nThe endpoint stopped responding at claim {i}/{len(claims)} "
                f"({system}).\n"
                f"  {llm.base_url} refused the connection.\n"
                f"  {i - 1} calls in this system are cached and will replay instantly.\n"
                f"  Restart the server and run the same command again to resume."
            ) from exc
        print(f"  [{i}/{len(claims)}] {trajectories[-1].verdict:<22} {claim.claim[:58]}")

    labelled = [t for t in trajectories if t.gold_label]
    metrics = score(
        [t.gold_label for t in labelled],
        [t.verdict for t in labelled],
        unparseable=sum(1 for t in trajectories if not t.metadata.get("parsed", True)),
    )
    return trajectories, metrics.as_dict() | {"summary": metrics.summary()}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    every = load_averitec(args.data, args.split)
    claims = sample(every, args.limit, args.seed)
    if not claims:
        raise SystemExit(f"No claims loaded from {args.data}")
    scope = "the whole split" if len(claims) == len(every) else f"a seed-{args.seed} sample"
    print(f"[verimem] {len(claims)} of {len(every)} claims from {args.data} ({scope})")
    print(f"[verimem] gold labels: {label_counts(claims)}")

    if args.system == "all":
        systems = list(SYSTEMS)
    elif args.system == "both":
        systems = list(BASELINE_SYSTEMS)
    else:
        systems = [args.system]

    # Checked before the endpoint, the evaluator or any model is touched: a bad
    # invocation should cost seconds, not the whole setup.
    if args.experience_retrieval:
        if "crag" not in systems:
            raise SystemExit(
                "--experience-retrieval only applies to the crag system; the "
                "no-retrieval and plain-RAG baselines are defined without it, and "
                "silently adding it would make those rows something other than what "
                "the table calls them."
            )
        if not args.trajectories:
            raise SystemExit(
                "--experience-retrieval needs --trajectories. Retrieving from an empty "
                "store would report ExpRAG as having no effect."
            )

    verifier = load_verifier(args.verifier_adapter, base_only=args.verifier_base)
    if verifier is not None:
        if systems != ["crag"]:
            raise SystemExit(
                "--verifier-adapter only applies to the crag system: the no-retrieval and "
                "plain-RAG baselines are defined as the prompted model, and swapping the "
                "verifier under them would make the comparison two changes rather than one."
            )
        print(f"[verimem] fine-tuned verifier: {verifier.model}")

    scorer = None
    if "crag" in systems:
        if args.evaluator is None:
            raise SystemExit(
                "--evaluator is required for crag. Without it the pipeline would score "
                "passages with a lexical stub, and the CRAG row would measure word "
                "counting rather than the trained evaluator."
            )
        scorer = load_scorer(args.evaluator)

    source = None
    if {"plain_rag", "crag"} & set(systems):
        if args.evidence == "gold":
            print("[verimem] WARNING: --evidence gold is an ORACLE upper bound, not a baseline")
            source = GoldEvidenceSource(claims)
        elif not args.store:
            # A dry run is meant to answer "what will this cost" before the data exists,
            # so a missing store is only fatal once we are actually about to call.
            if not args.dry_run:
                raise SystemExit("--store is required for plain_rag (or use --evidence gold)")
            print("[verimem] no --store given; plain_rag would run with no evidence")
        else:
            source = open_store(args.store)
            print(f"[verimem] store: {type(source).__name__} from {args.store}")
            missing = [c.id for c in claims if not source.has(str(c.id))]
            if len(missing) == len(claims):
                # Always a wrong path or the wrong store layout, never a real result.
                raise SystemExit(
                    f"No documents for ANY of the {len(claims)} claims in {args.store}. "
                    "Plain RAG would run with no evidence at all, which is not a baseline. "
                    "Check the path and the file layout."
                )
            if missing:
                print(
                    f"[verimem] WARNING: {len(missing)}/{len(claims)} claims have no documents; "
                    f"they will be verified with no evidence",
                    file=sys.stderr,
                )

    llm = LLMClient()
    total = sum(llm.estimate(len(claims), AVG_INPUT_TOKENS[s], AVG_OUTPUT_TOKENS) for s in systems)
    free = is_free(llm.base_url)
    print(f"[verimem] endpoint {llm.base_url} ({llm.provider}), model {llm.model}")
    print(f"[verimem] estimated cost: ${total:.4f}" + ("  (local endpoint, free)" if free else ""))

    n_calls = len(claims) * len(systems)
    if n_calls > 100:
        # Cached calls return instantly, so this is a worst case, not a prediction.
        print(
            f"[verimem] {n_calls} calls; up to ~{n_calls * SECONDS_PER_CALL / 3600:.1f} h "
            f"at {SECONDS_PER_CALL}s/call if none are cached. Ctrl-C now if that is not "
            f"what you meant."
        )

    if args.dry_run:
        print("[verimem] --dry-run: stopping before any call.")
        return 0
    if not free and not args.yes and total > 0:
        if input(f"Spend up to ${total:.4f}? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1

    cfg = RunConfig(
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        top_k_evidence=args.k,
        experience_retrieval=args.experience_retrieval,
        gate_experience=args.gate_experience,
        source_trust=args.source_trust,
        verifier="finetuned" if args.verifier_adapter else "prompted",
    )
    cfg.api_model, cfg.api_base_url = llm.model, llm.base_url

    memory = None
    if args.experience_retrieval:
        past = read_jsonl(args.trajectories)
        trust = BetaTrust()
        for t in past:
            trust.observe(t)
        memory = ExperienceStore(
            past,
            scorer=scorer,
            trust=trust,
            gate_threshold=cfg.gate_threshold,
            trust_floor=cfg.trust_floor,
        )
        print(
            f"[verimem] experience: {len(memory)} past verifications, k={cfg.k_trajectories}"
            f"  gate={cfg.gate_experience} trust={cfg.source_trust}"
        )
    if "crag" in systems:
        print(
            f"[verimem] crag thresholds: upper={cfg.crag_upper_threshold} "
            f"lower={cfg.crag_lower_threshold}"
        )

    out = run_dir(args.name or f"{args.system}-{args.split}")
    cfg.save(out)
    all_metrics = {}
    for system in systems:
        print(f"\n[verimem] === {system} ===")
        trajectories, metrics = run_one(
            system, claims, llm, source, args, scorer, cfg, verifier=verifier, memory=memory
        )
        all_metrics[system] = metrics
        write_jsonl(trajectories, out / f"{system}_trajectories.jsonl")
        print(metrics["summary"])

    all_metrics["usage"] = llm.usage.as_dict()
    announce_metrics(out, all_metrics)
    (out / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    _write_notes(out, args, llm, all_metrics, systems)
    print(f"\n[verimem] wrote {out}")
    print(f"[verimem] usage: {llm.usage.as_dict()}")
    return 0


def _write_notes(out: Path, args, llm, metrics: dict, systems: list[str]) -> None:
    rows = "\n".join(
        f"| {s} | {metrics[s]['n']} | {metrics[s]['accuracy']:.3f} | "
        f"{metrics[s]['macro_f1']:.3f} | {metrics[s]['unparseable']} |"
        for s in systems
    )
    (out / "notes.md").write_text(
        f"""# {out.name}

Split **{args.split}**, {args.limit} claims, seed {args.seed}, k={args.k}, evidence from
`{args.evidence}`. Teacher `{llm.model}` at `{llm.base_url}`.

| system | n | accuracy | macro-F1 | unparseable |
|---|---|---|---|---|
{rows}

Usage: {llm.usage.as_dict()}

To fill in: do the failures cluster on one verdict? Is the retriever or the verifier at
fault -- re-run plain_rag with `--evidence gold` to separate them.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
