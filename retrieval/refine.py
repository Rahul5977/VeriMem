"""Knowledge refinement: decompose into strips, score, recompose (P1.5).

CRAG's refinement step exists because a retrieved passage is rarely uniformly useful.
Most of it is context, a sentence or two actually bears on the claim, and feeding the
whole thing to the verifier both wastes context budget and buries the signal.

So a passage is split into strips, each strip is scored by the same evaluator that
graded the passage, and only the best survive into the recomposed text. The strips keep
their source domain, because C3's trust model needs to know which publisher a surviving
sentence came from -- and because a verdict that cites "[bbc.co.uk]" is auditable in a
way that an anonymous blob of text is not.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from evaluator.score import RelevanceScorer
from experience.format import Evidence

# Split on sentence boundaries, but not on the common abbreviations that would shatter
# a sentence mid-citation.
_ABBREV = (
    r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bSt)(?<!\bNo)(?<!\bvs)(?<!\bU\.S)(?<!\be\.g)(?<!\bi\.e)"
)
_SPLIT = re.compile(rf"{_ABBREV}(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

MIN_STRIP_CHARS = 25


@dataclass
class Strip:
    """One sentence-level unit of evidence, with where it came from and how it scored."""

    text: str
    url: str
    domain: str
    score: float = 0.0

    def render(self) -> str:
        return f"[{self.domain or 'unknown'}] {self.text}"


def split_strips(text: str, min_chars: int = MIN_STRIP_CHARS) -> list[str]:
    """Sentence strips. Fragments shorter than `min_chars` are navigation, not evidence."""
    out = []
    for piece in _SPLIT.split(text or ""):
        piece = " ".join(piece.split())
        if len(piece) >= min_chars:
            out.append(piece)
    return out


def strips_from_evidence(evidence: Sequence[Evidence]) -> list[Strip]:
    return [
        Strip(text=t, url=e.url, domain=e.domain) for e in evidence for t in split_strips(e.text)
    ]


def refine(
    claim: str,
    evidence: Sequence[Evidence],
    scorer: RelevanceScorer,
    top_k: int = 5,
    min_score: float | None = None,
) -> list[Strip]:
    """Decompose, score every strip against the claim, keep the best `top_k`.

    Returns strips in score order. `min_score` additionally drops strips the evaluator
    actively judges irrelevant, which is what makes refinement capable of returning
    *nothing* -- the honest answer when retrieval found only noise, and the signal the
    router needs to send a claim down the rewrite branch.
    """
    strips = strips_from_evidence(evidence)
    if not strips:
        return []
    for strip, value in zip(strips, scorer.score(claim, [s.text for s in strips]), strict=True):
        strip.score = value
    strips.sort(key=lambda s: -s.score)
    if min_score is not None:
        strips = [s for s in strips if s.score >= min_score]
    return strips[:top_k]


def recompose(strips: Sequence[Strip]) -> str:
    """The refined evidence block that goes to the verifier."""
    if not strips:
        return "(no relevant evidence found)"
    return "\n".join(f"{i}. {s.render()}" for i, s in enumerate(strips, start=1))


def as_evidence(strips: Sequence[Strip]) -> list[Evidence]:
    """Refined strips back in the trajectory format, so a run stays storable."""
    return [
        Evidence(
            url=s.url, domain=s.domain, text=s.text, grade=round(s.score, 4), retrieved_by="refined"
        )
        for s in strips
    ]


@dataclass
class TrustFilterStats:
    """What trust did to the evidence, so a run can report the cost of the defence."""

    considered: int = 0
    kept: int = 0
    dropped: int = 0
    poisoned_considered: int = 0
    poisoned_kept: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "strips_considered": self.considered,
            "strips_kept": self.kept,
            "strips_dropped_by_trust": self.dropped,
            "poisoned_strips_considered": self.poisoned_considered,
            "poisoned_strips_kept": self.poisoned_kept,
            "poison_pass_rate": round(self.poisoned_kept / self.poisoned_considered, 4)
            if self.poisoned_considered
            else None,
        }


def filter_by_trust(
    strips: Sequence[Strip],
    trust: object,
    floor: float = 0.3,
    fake_urls: frozenset[str] | set[str] = frozenset(),
    stats: TrustFilterStats | None = None,
    keep_at_least: int = 1,
) -> list[Strip]:
    """C3 on the evidence path: drop strips from publishers below the trust floor.

    This is the defence source poisoning has to be met with. A fabricated *page* is
    written about the target claim in detail with the stance the adversary wants, so it
    is genuinely relevant and the evaluator scores it high -- correctly. Relevance is not
    truthfulness, and CRAG's grading only measures the former, so C2 cannot help here.
    What distinguishes the fake is who published it.

    `keep_at_least` is a floor on the floor. Dropping every strip would hand the verifier
    an empty evidence block, and a verdict of "Not enough evidence" on no evidence is not
    a defence -- it is the attack succeeding by a different route. When nothing clears the
    threshold the most-trusted strips are kept and the verifier sees weak evidence rather
    than none.
    """
    stats = stats if stats is not None else TrustFilterStats()
    if not strips:
        return []

    scored = []
    for strip in strips:
        stats.considered += 1
        poisoned = strip.url in fake_urls
        if poisoned:
            stats.poisoned_considered += 1
        scored.append((strip, trust.trust(strip.domain), poisoned))  # type: ignore[attr-defined]

    kept = [(s, t, p) for s, t, p in scored if t >= floor]
    if not kept:
        # Ranked by trust, then by the evaluator's own score, so the fallback is the best
        # available evidence rather than an arbitrary slice.
        scored.sort(key=lambda row: (-row[1], -row[0].score))
        kept = scored[:keep_at_least]

    for _, _, poisoned in kept:
        stats.kept += 1
        if poisoned:
            stats.poisoned_kept += 1
    stats.dropped += stats.considered - len(kept)
    return [s for s, _, _ in kept]
