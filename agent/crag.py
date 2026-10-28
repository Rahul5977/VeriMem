"""CRAG: grade the retrieved evidence, then pick an action (P1.6, P1.7, P1.8).

The whole point of CRAG is that retrieval sometimes fails, and a system that cannot
notice its own retrieval failing will confidently verify a claim against irrelevant
text. So the evaluator grades what came back, and two thresholds turn that grade into
one of three actions:

    score >= upper   CORRECT    the evidence is good; refine it and answer
    score <= lower   INCORRECT  the evidence is wrong; rewrite the query and search again
    otherwise        AMBIGUOUS  do both and let the verifier see the union

**Why this matters for our numbers specifically.** The plain-RAG baseline scored 0.000
F1 on "Conflicting evidence" and 0.093 on "Not enough evidence" -- it is effectively a
Supported/Refuted classifier. Those are exactly the two outcomes that require noticing
something about the *evidence* rather than the claim: that sources disagree, or that
nothing retrieved actually bears on it. Routing is the mechanism that can see that.

**Search, here, means the knowledge store.** Rule 5 keeps benchmark runs off the live
web, so the "web search" branch re-retrieves from the pool with a rewritten query. That
is a real deviation from the paper and is reported as one: we test whether rewriting
helps retrieval, not whether the open web does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agent.baselines import SYSTEM, parse_verdict
from core.config import RunConfig
from core.llm import LLMClient
from eval.datasets import Claim
from evaluator.score import RelevanceScorer
from experience.format import Step, Trajectory
from experience.store import ExperienceStore
from retrieval.refine import Strip, as_evidence, recompose, refine
from retrieval.store import EvidenceSource

Action = Literal["correct", "incorrect", "ambiguous"]

REWRITE_SYSTEM = (
    "You turn a claim into a short web search query that would find evidence about it.\n"
    "Reply with the query only: no quotes, no explanation, no more than 12 words."
)

DECOMPOSE_SYSTEM = (
    "Break the claim into 2-3 short, independent search questions whose answers would "
    "together settle whether it is true.\n"
    "One question per line. No numbering, no bullets, no explanation."
)


@dataclass
class CragResult:
    verdict: str
    rationale: str
    action: Action
    grade: float
    strips: list[Strip] = field(default_factory=list)
    rewritten_query: str | None = None
    steps: list[Step] = field(default_factory=list)
    parsed: bool = True
    n_memory: int = 0


def grade_evidence(claim: str, evidence, scorer: RelevanceScorer) -> float:
    """One number for "how good is this retrieval".

    The maximum over passages, not the mean: one genuinely relevant document is enough
    to answer a claim, and averaging lets four irrelevant passages drown it out and
    send a perfectly answerable claim down the rewrite branch.
    """
    if not evidence:
        return -1.0
    return max(scorer.score(claim, [e.text for e in evidence]), default=-1.0)


def route(grade: float, upper: float, lower: float) -> Action:
    if grade >= upper:
        return "correct"
    if grade <= lower:
        return "incorrect"
    return "ambiguous"


def rewrite_query(claim: str, llm: LLMClient, max_tokens: int = 40) -> str:
    """Turn the claim into a search query. Cached like every other model call."""
    reply = llm.complete(f"Claim: {claim}", system=REWRITE_SYSTEM, max_tokens=max_tokens)
    query = " ".join(reply.text.strip().strip('"').split())
    # A model that ignores the instruction and explains itself would otherwise send a
    # paragraph to BM25, where it retrieves noise.
    return query[:200] or claim


def decompose_claim(claim: str, llm: LLMClient, n: int = 3, max_tokens: int = 80) -> list[str]:
    """Sub-questions whose answers would settle the claim.

    A reworded claim retrieves nearly the same documents as the claim did, which is why
    the rewrite branch was worth so little: it reshuffled rather than reached. Separate
    sub-questions have different content words, so they pull different documents out of
    the same corpus.

    This also mirrors how AVeriTeC was annotated -- each claim comes with the questions a
    fact-checker asked -- so it reproduces the structure of the task rather than guessing.
    """
    reply = llm.complete(f"Claim: {claim}", system=DECOMPOSE_SYSTEM, max_tokens=max_tokens)
    out = []
    for line in (reply.text or "").splitlines():
        line = line.strip().lstrip("-*0123456789.) ").strip()
        if len(line) > 8:
            out.append(line[:200])
    return out[:n] or [claim]


def run_crag(
    claim: Claim,
    llm: LLMClient,
    source: EvidenceSource,
    scorer: RelevanceScorer,
    cfg: RunConfig | None = None,
    force_action: Action | None = None,
    memory: ExperienceStore | None = None,
) -> CragResult:
    """Retrieve, grade, route, refine and/or re-search, then verify.

    `force_action` overrides the router. Threshold tuning (P1.7) needs each branch's
    verdict for the same claim, so it can sweep the grid without re-running the
    pipeline at every grid point. Nothing else should pass it.
    """
    cfg = cfg or RunConfig()
    steps: list[Step] = []

    evidence = source.retrieve(str(claim.id), claim.claim, k=cfg.top_k_evidence)
    steps.append(Step("retrieve", f"top-{cfg.top_k_evidence}", score=float(len(evidence))))

    grade = grade_evidence(claim.claim, evidence, scorer)
    action = force_action or route(grade, cfg.crag_upper_threshold, cfg.crag_lower_threshold)
    steps.append(Step("grade", "max strip-level relevance", score=round(grade, 4)))
    steps.append(Step("route", action, score=round(grade, 4)))

    strips: list[Strip] = []
    rewritten: str | None = None

    if action in ("correct", "ambiguous"):
        strips = refine(claim.claim, evidence, scorer, top_k=cfg.top_k_strips)
        steps.append(Step("refine", f"{len(strips)} strips kept"))

    if action in ("incorrect", "ambiguous"):
        queries = decompose_claim(claim.claim, llm, cfg.n_subquestions)
        rewritten = " | ".join(queries)
        steps.append(Step("rewrite", rewritten, score=float(len(queries))))

        # Retrieve per sub-question and deeper than the first pass, so this branch can
        # surface documents the original query never reached. Rule 5: the knowledge
        # store, not the live web.
        per_query = max(1, cfg.rewrite_top_k // len(queries))
        extra, seen = [], {(e.url, e.text[:80]) for e in evidence}
        for query in queries:
            for hit in source.retrieve(str(claim.id), query, k=per_query):
                key = (hit.url, hit.text[:80])
                if key not in seen:
                    seen.add(key)
                    extra.append(hit)
        steps.append(
            Step("search", f"knowledge store, {len(extra)} new passages", score=float(len(extra)))
        )
        more = refine(claim.claim, extra, scorer, top_k=cfg.top_k_strips)
        seen = {s.text for s in strips}
        strips += [s for s in more if s.text not in seen]
        strips.sort(key=lambda s: -s.score)
        strips = strips[: cfg.top_k_strips]

    # Experience retrieval (ExpRAG, and the surface C1 attacks). Off unless the flag is
    # set, so a run with every flag off reproduces plain CRAG exactly (rule 4).
    experience = ""
    n_memory = 0
    if memory is not None and cfg.experience_retrieval:
        retrieved = memory.retrieve(
            claim.claim,
            claim_id=str(claim.id),
            k=cfg.k_trajectories,
            gate=cfg.gate_experience,
            use_trust=cfg.source_trust,
        )
        n_memory = len(retrieved)
        experience = memory.render(retrieved, token_cap=cfg.trajectory_token_cap)
        steps.append(Step("experience", f"{n_memory} past verifications", score=float(n_memory)))

    block = recompose(strips)
    prompt = f"Claim: {claim.claim}\n\nEvidence:\n{block}"
    if experience:
        prompt = f"{experience}\n\n{prompt}"
    reply = llm.complete(prompt, system=SYSTEM, max_tokens=200)
    parsed = parse_verdict(reply.text)
    steps.append(Step("verify", f"verdict from {len(strips)} strips"))

    return CragResult(
        verdict=parsed.verdict,
        rationale=parsed.rationale,
        action=action,
        grade=grade,
        strips=strips,
        rewritten_query=rewritten,
        steps=steps,
        parsed=parsed.parsed,
        n_memory=n_memory,
    )


def crag(
    claim: Claim,
    llm: LLMClient,
    source: EvidenceSource,
    scorer: RelevanceScorer,
    cfg: RunConfig | None = None,
    memory: ExperienceStore | None = None,
) -> Trajectory:
    """CRAG as a `Trajectory`, so it stores, trains and evaluates like everything else."""
    result = run_crag(claim, llm, source, scorer, cfg, memory=memory)
    return Trajectory(
        claim=claim.claim,
        verdict=result.verdict,
        rationale=result.rationale,
        evidence=as_evidence(result.strips),
        steps=result.steps,
        claim_id=str(claim.id),
        gold_label=claim.gold_label,
        action=result.action,
        source="crag",
        dataset="averitec",
        split=str(claim.metadata.get("split", "dev")),
        metadata={
            "parsed": result.parsed,
            "grade": round(result.grade, 4),
            "rewritten_query": result.rewritten_query,
            "n_memory": result.n_memory,
        },
    )
