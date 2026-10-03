"""Build SFT examples from trajectories -- plain (P2.9) and retrieval-augmented (P2.10).

Two modes, one prompt shape, so the only difference measured at G2 is whether past
trajectories were in the context:

    plain      system + (claim, evidence)                  -> verdict
    retrieval  system + k past trajectories + (claim, ...)  -> verdict

Examples are emitted in TRL's conversational prompt/completion form, so the model's own
chat template is applied at training time and nothing here hard-codes Qwen tokens.

Pure Python -- no torch -- so the dataset can be built and inspected on the Mac, and the
GPU box only has to train. The token budget is enforced here rather than left to the
trainer to truncate, because a silently truncated example loses its answer.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from core.config import RunConfig
from experience.format import TokenCounter, Trajectory, estimate_tokens
from training.neighbors import LexicalRetriever, Retriever

Mode = Literal["plain", "retrieval"]

SYSTEM_PROMPT = (
    "You are a fact-checking verifier. Given a claim and retrieved evidence, decide "
    "whether the evidence supports the claim, refutes it, conflicts, or is insufficient.\n"
    "Answer with exactly one of: Supported, Refuted, Conflicting evidence, "
    "Not enough evidence.\n"
    "Reply in this form:\nVerdict: <label>\nReasoning: <one or two sentences>"
)

# Room left for the model's own answer, so a long prompt never crowds out the target.
COMPLETION_RESERVE_TOKENS = 220
# Slack for the chat template's role markers and separators, which we cannot see here.
TEMPLATE_OVERHEAD_TOKENS = 80


@dataclass
class SFTExample:
    prompt: list[dict[str, str]]
    completion: list[dict[str, str]]
    trajectory_id: str
    n_retrieved: int = 0
    est_tokens: int = 0
    trimmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "completion": self.completion}


@dataclass
class DatasetStats:
    n_examples: int = 0
    n_skipped: int = 0
    n_trimmed: int = 0
    n_with_retrieval: int = 0
    max_tokens: int = 0
    mean_tokens: float = 0.0
    verdict_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples,
            "n_skipped": self.n_skipped,
            "n_trimmed": self.n_trimmed,
            "n_with_retrieval": self.n_with_retrieval,
            "max_tokens": self.max_tokens,
            "mean_tokens": round(self.mean_tokens, 1),
            "verdict_counts": self.verdict_counts,
        }


def _user_message(claim: str, evidence_block: str, context_block: str = "") -> str:
    parts = []
    if context_block:
        parts.append(f"Past verifications that may help:\n\n{context_block}")
    parts.append(f"Claim: {claim}")
    parts.append(f"Evidence:\n{evidence_block}")
    return "\n\n".join(parts)


def build_example(
    trajectory: Trajectory,
    retrieved: Sequence[Trajectory] = (),
    cfg: RunConfig | None = None,
    count: TokenCounter = estimate_tokens,
) -> SFTExample:
    """One training example, shrunk until it fits `cfg.max_seq_length`.

    Shrinking order is deliberate: drop retrieved trajectories before squeezing the
    current claim's evidence, because the evidence is what the verdict must be grounded
    in. The target answer is never truncated.
    """
    cfg = cfg or RunConfig()
    answer = trajectory.render_answer()
    budget = (
        cfg.max_seq_length
        - COMPLETION_RESERVE_TOKENS
        - TEMPLATE_OVERHEAD_TOKENS
        - count(SYSTEM_PROMPT)
        - count(trajectory.claim)
    )
    if count(answer) > COMPLETION_RESERVE_TOKENS:
        budget -= count(answer) - COMPLETION_RESERVE_TOKENS

    kept = list(retrieved)
    trimmed = False
    while True:
        evidence_budget = budget - len(kept) * cfg.trajectory_token_cap
        if evidence_budget >= 256 or not kept:
            break
        kept.pop()  # weakest neighbour goes first; the retriever returns them ranked
        trimmed = True

    context_block = "\n\n---\n\n".join(
        t.render_for_context(cfg.trajectory_token_cap, count) for t in kept
    )
    evidence_block = trajectory.render_evidence(max(budget - count(context_block), 128), count)
    user = _user_message(trajectory.claim, evidence_block, context_block)

    est = count(SYSTEM_PROMPT) + count(user) + count(answer) + TEMPLATE_OVERHEAD_TOKENS
    return SFTExample(
        prompt=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        completion=[{"role": "assistant", "content": answer}],
        trajectory_id=trajectory.id,
        n_retrieved=len(kept),
        est_tokens=est,
        trimmed=trimmed or len(kept) < len(retrieved),
    )


def build_dataset(
    trajectories: Sequence[Trajectory],
    mode: Mode = "plain",
    cfg: RunConfig | None = None,
    retriever: Retriever | None = None,
    keep_incorrect: bool = False,
    count: TokenCounter = estimate_tokens,
) -> tuple[list[SFTExample], DatasetStats]:
    """Turn trajectories into SFT examples.

    By default only trajectories whose verdict matched the gold label are trained on;
    failures are kept separately (P2.4) for the error analysis, not used as targets.
    Trajectories with no gold label are always kept, so an unlabelled pilot still works.
    """
    cfg = cfg or RunConfig()
    k = cfg.sft_k_trajectories if mode == "retrieval" else 0
    if mode == "retrieval" and retriever is None:
        retriever = LexicalRetriever(trajectories).retrieve

    examples: list[SFTExample] = []
    stats = DatasetStats()
    for traj in trajectories:
        if traj.correct is False and not keep_incorrect:
            stats.n_skipped += 1
            continue
        neighbours = retriever(traj, k) if (k and retriever) else []
        example = build_example(traj, neighbours, cfg, count)
        examples.append(example)
        stats.n_examples += 1
        stats.n_trimmed += int(example.trimmed)
        stats.n_with_retrieval += int(example.n_retrieved > 0)
        stats.max_tokens = max(stats.max_tokens, example.est_tokens)
        stats.verdict_counts[traj.verdict] = stats.verdict_counts.get(traj.verdict, 0) + 1

    if examples:
        stats.mean_tokens = sum(e.est_tokens for e in examples) / len(examples)
    return examples, stats


def write_dataset(examples: Sequence[SFTExample], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    return path
