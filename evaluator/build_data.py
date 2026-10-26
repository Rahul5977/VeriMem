"""Build relevance training data for the CRAG evaluator (P1.1).

The evaluator learns to score how relevant a retrieved passage is to a claim. CRAG's
original training data came from existing relevance labels; AVeriTeC gives us the same
thing for free, because every claim's annotation names the source URLs its evidence came
from. So a passage's label is decided by where it came from, not by asking a model:

* **positive** -- the passage comes from a URL the annotators cited for this claim
* **hard negative** -- BM25 ranks it highly for this claim, but no annotator cited it.
  These are the confusions the evaluator actually has to resolve at inference time, and
  they are what makes this dataset worth building rather than sampling at random.
* **easy negative** -- a passage drawn from elsewhere in the pool entirely

No model is involved, so there are no label-noise questions beyond the annotation itself,
and it costs nothing.

**Train on the train split.** The evaluator feeds the full system, so an evaluator fitted
on dev claims would have seen the test set by the time the system is scored on dev
(rule 5). `--data` therefore defaults to nothing and must be given explicitly.

The 90/10 split is **by claim**, never by pair: a claim's positives and its hard
negatives are highly correlated, so splitting pairs would put near-duplicates on both
sides and overstate dev accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.paths import DATA_DIR
from eval.datasets import Claim, load_averitec
from retrieval.bm25 import BM25Index
from retrieval.store import PooledKnowledgeStore

POSITIVE, HARD_NEGATIVE, EASY_NEGATIVE = "positive", "hard_negative", "easy_negative"


@dataclass
class Pair:
    claim: str
    passage: str
    url: str
    label: int  # 1 relevant, 0 not
    kind: str
    claim_id: str
    split: str = "train"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gold_urls(claim: Claim) -> set[str]:
    """Every source URL the annotators cited for this claim."""
    out = set()
    for question in claim.questions:
        for answer in question.get("answers", []) or []:
            for key in ("cached_source_url", "source_url"):
                if answer.get(key):
                    out.add(answer[key])
    return out


def build_pairs(
    claims: Sequence[Claim],
    store: PooledKnowledgeStore,
    hard_per_claim: int = 3,
    easy_per_claim: int = 1,
    max_positives: int = 3,
    seed: int = 13,
) -> tuple[list[Pair], dict[str, int]]:
    """One pass over the claims, emitting labelled (claim, passage) pairs."""
    rng = random.Random(seed)
    store._build()  # noqa: SLF001 - we need the passage table, not just search
    by_url: dict[str, list[tuple[str, str]]] = {}
    for pid, (url, text) in store._passages.items():  # noqa: SLF001
        by_url.setdefault(url, []).append((pid, text))
    all_urls = list(by_url)

    pairs: list[Pair] = []
    stats = {
        "claims": 0,
        "claims_without_gold_in_pool": 0,
        POSITIVE: 0,
        HARD_NEGATIVE: 0,
        EASY_NEGATIVE: 0,
    }

    for claim in claims:
        gold = gold_urls(claim)
        present = [u for u in gold if u in by_url]
        if not present:
            # The annotated pages failed to fetch; nothing to call relevant.
            stats["claims_without_gold_in_pool"] += 1
            continue
        stats["claims"] += 1

        for url in present:
            # Rank the cited page's own passages against the claim rather than taking
            # the first few. The top of a page is masthead, navigation and byline; the
            # part that actually addresses the claim is somewhere in the middle. Taking
            # position 0-2 paired boilerplate-as-relevant against BM25-selected
            # on-topic text-as-irrelevant, which teaches the inverse of the real task.
            page = by_url[url]
            index = BM25Index((pid, text) for pid, text in page)
            ranked = index.search(claim.claim, k=max_positives)
            chosen = [pid for pid, _ in ranked] or [pid for pid, _ in page[:max_positives]]
            texts = dict(page)
            for pid in chosen[:max_positives]:
                pairs.append(Pair(claim.claim, texts[pid], url, 1, POSITIVE, str(claim.id)))
                stats[POSITIVE] += 1

        # Hard negatives: top BM25 hits that no annotator cited.
        taken = 0
        for ev in store.retrieve(str(claim.id), claim.claim, k=hard_per_claim + len(present) + 5):
            if ev.url in gold or taken >= hard_per_claim:
                continue
            pairs.append(Pair(claim.claim, ev.text, ev.url, 0, HARD_NEGATIVE, str(claim.id)))
            stats[HARD_NEGATIVE] += 1
            taken += 1

        # Easy negatives: anywhere else in the pool.
        for _ in range(easy_per_claim):
            url = rng.choice(all_urls)
            if url in gold:
                continue
            pairs.append(
                Pair(claim.claim, rng.choice(by_url[url])[1], url, 0, EASY_NEGATIVE, str(claim.id))
            )
            stats[EASY_NEGATIVE] += 1

    return pairs, stats


def split_by_claim(
    pairs: Sequence[Pair], dev_fraction: float = 0.1, seed: int = 13
) -> tuple[list[Pair], list[Pair]]:
    """90/10 split on claims, so no claim contributes to both sides (P1.2)."""
    claim_ids = sorted({p.claim_id for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(claim_ids)
    n_dev = max(1, int(len(claim_ids) * dev_fraction))
    dev_ids = set(claim_ids[:n_dev])
    train, dev = [], []
    for p in pairs:
        (dev if p.claim_id in dev_ids else train).append(
            Pair(**{**p.to_dict(), "split": "dev" if p.claim_id in dev_ids else "train"})
        )
    return train, dev


def write_jsonl(pairs: Sequence[Pair], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evaluator.build_data")
    p.add_argument("--data", type=Path, required=True, help="AVeriTeC split (use train.json)")
    p.add_argument("--pool", type=Path, required=True, help="the matching evidence pool")
    p.add_argument("--out", type=Path, default=DATA_DIR / "evaluator")
    p.add_argument("--hard", type=int, default=3, help="hard negatives per claim")
    p.add_argument("--easy", type=int, default=1, help="easy negatives per claim")
    p.add_argument("--max-positives", type=int, default=3)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args(argv)

    if args.data.stem == "dev":
        print(
            "[verimem] WARNING: building evaluator data from the dev split. The evaluator "
            "feeds the system that is scored on dev, so this leaks (rule 5). Use train.json."
        )

    claims = load_averitec(args.data, args.data.stem)
    store = PooledKnowledgeStore(args.pool)
    pairs, stats = build_pairs(claims, store, args.hard, args.easy, args.max_positives, args.seed)
    if not pairs:
        raise SystemExit(
            f"No pairs built. Does {args.pool} actually contain the pages cited by "
            f"{args.data}? A dev pool cannot label train claims."
        )

    train, dev = split_by_claim(pairs, seed=args.seed)
    write_jsonl(train, args.out / "relevance_train.jsonl")
    write_jsonl(dev, args.out / "relevance_dev.jsonl")

    pos = sum(1 for x in pairs if x.label == 1)
    print(
        f"[verimem] {stats['claims']} claims usable, "
        f"{stats['claims_without_gold_in_pool']} skipped (gold pages not in the pool)"
    )
    print(
        f"[verimem] {len(pairs)} pairs: {pos} positive / {len(pairs) - pos} negative "
        f"({stats[HARD_NEGATIVE]} hard, {stats[EASY_NEGATIVE]} easy)"
    )
    print(f"[verimem] split by claim -> {len(train)} train / {len(dev)} dev")
    print(f"[verimem] wrote {args.out}/relevance_{{train,dev}}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
