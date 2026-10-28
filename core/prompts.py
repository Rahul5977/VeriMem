"""The verifier prompt, defined once.

It used to be defined twice -- `agent.baselines.SYSTEM` for inference and
`training.data.SYSTEM_PROMPT` for building SFT examples -- and the two drifted: the
training copy said "Given a claim and retrieved evidence, decide", the inference copy
just "Decide". A model fine-tuned on one string and then served the other is being asked
a question it was not trained on. The effect is small and entirely self-inflicted, and
it is invisible in both files because neither one is wrong on its own.

So the prompt lives here, imported by both. This module holds **no dependencies beyond
the standard library** on purpose: `training.data` is deliberately torch-free so datasets
can be built and inspected on the Mac, and importing the agent package to reach a string
constant would undo that.

Changing the system prompt invalidates the LLM cache for every verify call and makes
existing adapters slightly stale, so treat it as a decision to log (rule 3's spirit),
not a wording tweak.
"""

from __future__ import annotations

VERDICTS: tuple[str, ...] = (
    "Supported",
    "Refuted",
    "Conflicting evidence",
    "Not enough evidence",
)

VERIFIER_SYSTEM = (
    "You are a fact-checking verifier. Given a claim and retrieved evidence, decide "
    "whether the evidence supports the claim, refutes it, conflicts, or is insufficient.\n"
    "Answer with exactly one of: " + ", ".join(VERDICTS) + ".\n"
    "Reply in this form:\nVerdict: <label>\nReasoning: <one or two sentences>"
)

EXPERIENCE_HEADER = "Past verifications that may help:"


def verifier_user_message(claim: str, evidence_block: str, context_block: str = "") -> str:
    """The user turn, identical at training time and inference time.

    The experience block goes first and the claim last so that the thing being judged is
    nearest the answer. Both callers must produce byte-identical text for the same
    inputs, which is why neither one formats this itself.
    """
    parts = []
    if context_block:
        parts.append(f"{EXPERIENCE_HEADER}\n\n{context_block}")
    parts.append(f"Claim: {claim}")
    parts.append(f"Evidence:\n{evidence_block}")
    return "\n\n".join(parts)


def prompt_fingerprint() -> str:
    """A short hash of the verifier prompt, stamped into every run's config.

    The prompt is part of the LLM cache key, so changing one word silently invalidates
    every cached verify call and makes every number already on disk unreproducible from
    the current code. That happened once: the prompt existed in two places, they drifted,
    and unifying them moved the ground under results that had already been reported.

    Recording the fingerprint does not prevent the change -- it makes it *visible*. A
    results folder whose fingerprint differs from today's was produced by a different
    prompt, and comparing the two as though they were the same experiment is the mistake
    this guards against.
    """
    import hashlib

    return hashlib.sha256(VERIFIER_SYSTEM.encode("utf-8")).hexdigest()[:12]
