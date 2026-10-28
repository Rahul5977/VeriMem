"""The experience store: retrieve past verifications as context (P2.6, C2).

ExpRAG's premise is that an agent's own history is useful context. This is the store
that makes that possible -- and, because anything retrievable is also attackable, it is
the surface C1's memory-poisoning study targets.

Three behaviours, switched by the config flags so that turning them all off reproduces
the no-memory baseline exactly (rule 4):

* **plain retrieval** -- top-k trajectories by similarity to the claim. ExpRAG as
  published, and the configuration H1 predicts is the most vulnerable.
* **evaluator gating (C2)** -- each retrieved trajectory is scored by the same evaluator
  that grades evidence, and low scorers are dropped. A fabricated trajectory asserts a
  verdict its evidence does not carry; the gate is the thing that can notice.
* **source trust (C3)** -- trajectories whose evidence leans on domains with low Beta
  trust are dropped. Poisoned records cite the same handful of misinformation domains
  repeatedly, so trust accumulates against them.

**Retrieval is BM25, not FAISS.** The plan called for FAISS over sentence-transformer
embeddings. BM25 is already shared with passage retrieval, needs no model on an 8 GB
machine, and is deterministic, which matters when the poisoning study has to compare
runs that differ only in the store's contents. It is a deviation and is reported as one;
nothing in C1 or C2 depends on which retriever is underneath, and `Retriever` is a
protocol so FAISS can be dropped in later without touching callers.

Self-exclusion (rule 5) is enforced here, not left to callers: a trajectory must never
be retrieved into its own verification.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from evaluator.score import RelevanceScorer
from experience.format import Trajectory, read_jsonl
from retrieval.bm25 import BM25Index


class TrustModel(Protocol):
    """C3's interface. `trust(domain)` returns a value in [0, 1]."""

    def trust(self, domain: str) -> float: ...


@dataclass
class Retrieved:
    """A retrieved trajectory, with why it survived or would not have."""

    trajectory: Trajectory
    similarity: float
    gate_score: float | None = None
    min_trust: float | None = None
    dropped_by: str | None = None

    @property
    def kept(self) -> bool:
        return self.dropped_by is None


@dataclass
class RetrievalStats:
    """What the store did, so a poisoning run can report why an attack worked."""

    considered: int = 0
    kept: int = 0
    dropped_by_gate: int = 0
    dropped_by_trust: int = 0
    poisoned_considered: int = 0
    poisoned_kept: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "kept": self.kept,
            "dropped_by_gate": self.dropped_by_gate,
            "dropped_by_trust": self.dropped_by_trust,
            "poisoned_considered": self.poisoned_considered,
            "poisoned_kept": self.poisoned_kept,
            "poison_pass_rate": round(self.poisoned_kept / self.poisoned_considered, 4)
            if self.poisoned_considered
            else None,
        }


def is_self(query_claim: str, query_claim_id: str | None, candidate: Trajectory) -> bool:
    """Rule 5. Matching on claim text as well as id, because a poisoned record carries
    the target's wording under a different id -- and because the same claim can be
    re-verified under a new trajectory id."""
    if query_claim_id is not None and candidate.claim_id == query_claim_id:
        return True
    return candidate.claim.strip().lower() == query_claim.strip().lower()


class ExperienceStore:
    """Past trajectories, retrievable as context and optionally filtered."""

    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        scorer: RelevanceScorer | None = None,
        trust: TrustModel | None = None,
        gate_threshold: float = 0.0,
        trust_floor: float = 0.3,
    ) -> None:
        self.trajectories = list(trajectories)
        self.scorer = scorer
        self.trust_model = trust
        self.gate_threshold = gate_threshold
        self.trust_floor = trust_floor
        self._by_id = {t.id: t for t in self.trajectories}
        self._index = BM25Index((t.id, t.claim) for t in self.trajectories)
        self.stats = RetrievalStats()

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def poisoned_count(self) -> int:
        return sum(1 for t in self.trajectories if t.poisoned)

    @classmethod
    def from_jsonl(cls, path: Path | str, **kwargs: Any) -> ExperienceStore:
        return cls(read_jsonl(path), **kwargs)

    def retrieve(
        self,
        claim: str,
        claim_id: str | None = None,
        k: int = 3,
        gate: bool = False,
        use_trust: bool = False,
    ) -> list[Retrieved]:
        """Top-k past trajectories for this claim, after any enabled filters.

        Over-fetches before filtering: gating that drops two of three results should
        still return three, or C2 would reduce the amount of context as well as its
        quality and the comparison would confound the two.
        """
        if k <= 0 or not self.trajectories:
            return []

        want = k * 4 if (gate or use_trust) else k
        hits = self._index.search(
            claim, k=want, exclude=lambda tid: is_self(claim, claim_id, self._by_id[tid])
        )
        candidates = [Retrieved(self._by_id[tid], round(score, 4)) for tid, score in hits]

        for item in candidates:
            self.stats.considered += 1
            if item.trajectory.poisoned:
                self.stats.poisoned_considered += 1

            if use_trust and self.trust_model is not None:
                domains = item.trajectory.domains
                item.min_trust = min(self.trust_model.trust(d) for d in domains) if domains else 1.0
                if item.min_trust < self.trust_floor:
                    item.dropped_by = "trust"
                    self.stats.dropped_by_trust += 1
                    continue

            if gate and self.scorer is not None:
                # Ask whether the trajectory's evidence supports *its own* verdict, not
                # whether the trajectory is relevant to the new claim.
                #
                # The first version asked the latter, and it was defeated: a poisoned
                # trajectory is written *about* the target claim, so it is genuinely
                # relevant and the evaluator scored it high -- correctly. Relevance is
                # not truthfulness. Scoring the evidence against the verdict it is
                # supposed to justify is a question fabricated evidence can actually
                # fail, because the fakes assert a conclusion their generic text does
                # not establish.
                item.gate_score = round(self._supports_own_verdict(item.trajectory), 4)
                if item.gate_score < self.gate_threshold:
                    item.dropped_by = "gate"
                    self.stats.dropped_by_gate += 1
                    continue

            self.stats.kept += 1
            if item.trajectory.poisoned:
                self.stats.poisoned_kept += 1

        kept = [c for c in candidates if c.kept][:k]
        return kept

    def _supports_own_verdict(self, trajectory: Trajectory) -> float:
        """Does this trajectory's evidence justify the verdict it reached?

        Scored as relevance between the trajectory's own claim-plus-verdict and each of
        its evidence passages, taking the best. A genuine verification cites text that
        bears on its conclusion; a fabricated one cites text asserting the conclusion
        without establishing it.
        """
        passages = [e.text for e in trajectory.evidence if e.text]
        if not passages:
            return -1.0  # a verification resting on no evidence justifies nothing
        question = f"{trajectory.claim} The verdict was: {trajectory.verdict}."
        return max(self.scorer.score(question, passages), default=-1.0)

    def render(self, retrieved: Sequence[Retrieved], token_cap: int = 1000) -> str:
        """The block that goes into the verifier's prompt."""
        if not retrieved:
            return ""
        blocks = [r.trajectory.render_for_context(token_cap) for r in retrieved]
        return "Past verifications that may help:\n\n" + "\n\n---\n\n".join(blocks)


@dataclass
class UniformTrust:
    """A stand-in trust model until C3 lands. Every domain is trusted equally.

    Named rather than implicit: a run that silently used uniform trust while claiming
    to test C3 would report that trust does not help, which is a false negative on H2.
    """

    value: float = 1.0

    def trust(self, domain: str) -> float:
        return self.value


@dataclass
class BetaTrust:
    """C3: Beta(alpha, beta) per domain, trust = alpha / (alpha + beta).

    **An unseen domain is not a trusted one.** The first version used Beta(1, 1), which
    puts a stranger at 0.5 -- above any floor low enough to be usable, so a cold-start
    adversary registering fresh domains walked straight through. Trust that only
    distrusts domains it has already watched fail cannot defend against a new attacker,
    which is the only kind worth defending against.

    Beta(1, 3) starts a stranger at 0.25: below a 0.3 floor, so an unknown domain must
    earn its way in rather than being believed by default. Two successful verifications
    lift it to 0.5; a domain with a real track record is unaffected.
    """

    alpha: dict[str, float] = field(default_factory=dict)
    beta: dict[str, float] = field(default_factory=dict)
    prior_alpha: float = 1.0
    prior_beta: float = 3.0

    def trust(self, domain: str) -> float:
        a = self.alpha.get(domain, self.prior_alpha)
        b = self.beta.get(domain, self.prior_beta)
        return a / (a + b)

    def update(self, domain: str, was_correct: bool) -> None:
        key = self.alpha if was_correct else self.beta
        prior = self.prior_alpha if was_correct else self.prior_beta
        key[domain] = key.get(domain, prior) + 1.0

    def observe(self, trajectory: Trajectory) -> None:
        """Credit or blame every domain a completed verification relied on."""
        if trajectory.correct is None:
            return
        for domain in trajectory.domains:
            self.update(domain, trajectory.correct)

    def as_dict(self) -> dict[str, Any]:
        return {d: round(self.trust(d), 4) for d in sorted(set(self.alpha) | set(self.beta))}
