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
