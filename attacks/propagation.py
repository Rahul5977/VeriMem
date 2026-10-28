"""Propagation: does a poisoned verdict contaminate claims the adversary never touched? (P3.8, C1)

Memory poisoning measures what happens to the claims under attack. This measures what
happens *next*, and it is the part that makes memory poisoning different in kind from
ordinary adversarial input.

**The mechanism.** A fabricated trajectory makes the agent misjudge claim A. That verdict
is then stored as experience -- and the stored record is *not* fabricated. It is a real
trajectory, produced by the real pipeline, citing real retrieved evidence, carrying a
wrong verdict. Every defence in the system is aimed at the wrong thing:

* C2 asks whether a trajectory's evidence supports its own verdict. A's evidence was
  retrieved honestly and the model genuinely concluded the wrong verdict from it, so the
  record is far more coherent than a fabrication.
* C3 asks who published the evidence. A cited ordinary publishers, because the fakes that
  misled it are no longer in the picture.

So the adversary's influence outlives the fabricated records that carried it. Removing
every fake from the store does not undo it. That is the finding this module exists to
test, and it is why "detect and delete the poison" is not a sufficient answer.

**Pairs.** A propagation pair is (A, B): A is a poisoning target, B is a different claim
close enough to A that A's trajectory is retrieved when B is verified. Without that
closeness there is no channel, so pairing is by the same BM25 index the experience store
uses -- not by topic labels, which would pair claims the retriever never actually
connects.

**What is measured.** B is verified three times against otherwise identical stores:

1. with A's *correct* trajectory in the store -- the control,
2. with A's *corrupted* trajectory in the store -- the propagation condition,
3. with no A trajectory at all -- to separate "A's record moved B" from "any neighbour
   moves B".

If B's verdict shifts toward A's corrupted stance in (2) but not (1), the poison has
propagated one hop beyond its target.

**Ethics (rule 10).** Nothing here fabricates content. The corrupted trajectories are
the agent's own output under attack, and they stay in run directories.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from attacks.memory import Target
from eval.datasets import Claim
from experience.format import Trajectory
from retrieval.bm25 import BM25Index


@dataclass
class Pair:
    """One (target, neighbour) pair and the channel strength between them."""

    target_id: str
    target_claim: str
    neighbour_id: str
    neighbour_claim: str
    neighbour_gold: str
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_pairs(
    claims: Sequence[Claim],
    targets: Sequence[Target],
    min_similarity: float = 0.15,
    per_target: int = 1,
) -> list[Pair]:
    """Pair each target with the nearest *different* claim that could retrieve it.

    Neighbours come from the same BM25 index the experience store uses, because the
    question is not "are these claims related" but "would A's trajectory actually be
    retrieved when B is verified". A topic-label pairing would include pairs with no
    retrieval channel and dilute the effect toward zero.

    **Similarity is relative, not absolute.** BM25 scores are unbounded and scale with
    corpus size and IDF, so a fixed cutoff silently finds no pairs on a small set and
    every pair on a large one. Each neighbour's score is divided by the target's own
    self-score -- the score the target gets against itself, which is the maximum
    achievable on this index -- giving a corpus-independent 0..1 fraction.

    A neighbour must be labelled: an unlabelled claim can show a verdict *change* but not
    a change against ground truth, and those are not the same finding.
    """
    by_id = {str(c.id): c for c in claims}
    labelled = [c for c in claims if c.gold_label]
    index = BM25Index((str(c.id), c.claim) for c in labelled)

    pairs: list[Pair] = []
    target_ids = {t.claim_id for t in targets}
    for target in targets:
        if target.claim_id not in by_id:
            continue

        # One unfiltered search gives both the denominator and the candidates: the
        # target scores highest against its own text.
        hits = index.search(target.claim, k=per_target + len(target_ids) + 8)
        self_score = next((sc for cid, sc in hits if cid == target.claim_id), 0.0)
        if self_score <= 0:
            continue

        taken = 0
        for cid, score in hits:
            if taken >= per_target:
                break
            # A claim already under direct attack cannot show *propagated* harm.
            if cid in target_ids or cid not in by_id:
                continue
            relative = score / self_score
            if relative < min_similarity:
                break  # hits are ranked, so nothing further can qualify
            neighbour = by_id[cid]
            pairs.append(
                Pair(
                    target_id=target.claim_id,
                    target_claim=target.claim,
                    neighbour_id=cid,
                    neighbour_claim=neighbour.claim,
                    neighbour_gold=neighbour.gold_label or "",
                    similarity=round(relative, 4),
                )
            )
            taken += 1
    return pairs


def corrupted_copy(trajectory: Trajectory, verdict: str) -> Trajectory:
    """The same trajectory, re-labelled with the verdict the attack produced.

    Used to build the control: the *only* difference between the control store and the
    propagation store must be A's verdict. Rebuilding A's trajectory from scratch would
    also change its evidence and rationale, and then a shift in B could not be attributed.

    `poisoned` stays False on purpose. This record is not fabricated -- it is the
    pipeline's own honest output, which is precisely why the defences do not catch it.
    Marking it poisoned would let the bookkeeping find what the system cannot.
    """
    data = trajectory.to_dict()
    data.pop("correct", None)
    data["verdict"] = verdict
    data["id"] = f"{trajectory.id}-corrupted"
    data["metadata"] = {
        **trajectory.metadata,
        "propagation": "verdict replaced to isolate the effect of the verdict alone",
        "original_verdict": trajectory.verdict,
    }
    return Trajectory.from_dict(data)


def write_pairs(pairs: Sequence[Pair], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_pairs(path: Path) -> list[Pair]:
    return [
        Pair(**json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
