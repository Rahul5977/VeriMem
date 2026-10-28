"""The two Phase 0 baselines (P0.11), and the verdict parser they share.

* **No retrieval** -- the model answers from parametric knowledge alone. The floor.
* **Plain RAG** -- top-k passages from the knowledge store, no grading, no refinement,
  no rewriting. This is the row CRAG has to beat in the G1 table, so it stays
  deliberately dumb: whatever BM25 returns goes straight into the prompt.

Both emit a `Trajectory`, the same object the agent and the SFT pipeline use, so a
baseline run is already in the right shape to be stored, re-graded or trained on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.llm import LLMClient
from core.prompts import VERIFIER_SYSTEM
from eval.datasets import Claim
from experience.format import VERDICTS, Step, Trajectory
from retrieval.store import EvidenceSource

# One definition, shared with training so the fine-tuned verifier is served the
# prompt it was trained on. See core.prompts.
SYSTEM = VERIFIER_SYSTEM

SYSTEM_NO_EVIDENCE = (
    "You are a fact-checking verifier. You have no retrieved evidence - judge the claim "
    "from what you already know, and say so when you cannot.\n"
    "Answer with exactly one of: Supported, Refuted, Conflicting evidence, "
    "Not enough evidence.\n"
    "Reply in this form:\nVerdict: <label>\nReasoning: <one or two sentences>"
)

_VERDICT_LINE = re.compile(r"verdict\s*[:\-]\s*(.+)", re.IGNORECASE)
# Longest first, so "Not enough evidence" is not shadowed by a shorter alternative.
_ALIASES: list[tuple[str, str]] = sorted(
    [
        ("conflicting evidence/cherrypicking", "Conflicting evidence"),
        ("conflicting evidence", "Conflicting evidence"),
        ("not enough evidence", "Not enough evidence"),
        ("insufficient evidence", "Not enough evidence"),
        ("cherrypicking", "Conflicting evidence"),
        ("conflicting", "Conflicting evidence"),
        ("not enough", "Not enough evidence"),
        ("supported", "Supported"),
        ("refuted", "Refuted"),
        ("support", "Supported"),
        ("refute", "Refuted"),
        ("true", "Supported"),
        ("false", "Refuted"),
        ("nei", "Not enough evidence"),
    ],
    key=lambda pair: -len(pair[0]),
)

FALLBACK_VERDICT = "Not enough evidence"


@dataclass
class ParsedVerdict:
    verdict: str
    rationale: str
    parsed: bool  # False means the fallback was used; counted in the metrics


def parse_verdict(text: str) -> ParsedVerdict:
    """Pull a verdict out of a model reply, tolerantly.

    An unparseable reply becomes "Not enough evidence" rather than being dropped -- that
    keeps gold and predictions aligned for scoring -- but `parsed=False` is recorded so
    the rate is reported instead of hidden inside the accuracy number.
    """
    body = text or ""
    rationale = ""
    match = re.search(r"reasoning\s*[:\-]\s*(.+)", body, re.IGNORECASE | re.DOTALL)
    if match:
        rationale = " ".join(match.group(1).split())

    line = _VERDICT_LINE.search(body)
    candidate = (line.group(1) if line else body).strip().lower()

    for alias, verdict in _ALIASES:
        if alias in candidate:
            return ParsedVerdict(verdict, rationale, True)
    # No verdict line; try the whole reply before giving up.
    whole = body.strip().lower()
    for alias, verdict in _ALIASES:
        if alias in whole:
            return ParsedVerdict(verdict, rationale or " ".join(body.split())[:400], True)
    return ParsedVerdict(FALLBACK_VERDICT, " ".join(body.split())[:400], False)


def _trajectory(claim: Claim, parsed: ParsedVerdict, evidence, steps, source: str) -> Trajectory:
    return Trajectory(
        claim=claim.claim,
        verdict=parsed.verdict,
        rationale=parsed.rationale,
        evidence=list(evidence),
        steps=list(steps),
        claim_id=str(claim.id),
        gold_label=claim.gold_label,
        source=source,
        dataset="averitec",
        split=str(claim.metadata.get("split", "dev")),
        metadata={"parsed": parsed.parsed},
    )


def no_retrieval(claim: Claim, llm: LLMClient, max_tokens: int = 200) -> Trajectory:
    """Baseline 1: no evidence at all."""
    reply = llm.complete(f"Claim: {claim.claim}", system=SYSTEM_NO_EVIDENCE, max_tokens=max_tokens)
    parsed = parse_verdict(reply.text)
    return _trajectory(
        claim,
        parsed,
        [],
        [Step("verify", "no retrieval; parametric knowledge only")],
        "baseline_no_retrieval",
    )


def plain_rag(
    claim: Claim,
    llm: LLMClient,
    source: EvidenceSource,
    k: int = 5,
    max_tokens: int = 200,
) -> Trajectory:
    """Baseline 2: top-k passages, straight into the prompt, no grading or rewriting."""
    evidence = source.retrieve(str(claim.id), claim.claim, k=k)
    if evidence:
        block = "\n\n".join(f"{i}. {e.render()}" for i, e in enumerate(evidence, start=1))
    else:
        block = "(no evidence retrieved)"
    reply = llm.complete(
        f"Claim: {claim.claim}\n\nEvidence:\n{block}", system=SYSTEM, max_tokens=max_tokens
    )
    parsed = parse_verdict(reply.text)
    return _trajectory(
        claim,
        parsed,
        evidence,
        [
            Step("retrieve", f"BM25 top-{k} from knowledge store", score=float(len(evidence))),
            Step("verify", "verdict produced without grading or refinement"),
        ],
        "baseline_plain_rag",
    )


SYSTEMS = {"no_retrieval": no_retrieval, "plain_rag": plain_rag}


def verdict_is_valid(verdict: str) -> bool:
    return verdict in VERDICTS
