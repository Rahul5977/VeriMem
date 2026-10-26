"""Relevance scoring at inference time (P1.5, P1.7).

CRAG's evaluator returns a graded score in [-1, 1] for a (claim, passage) pair. Two
thresholds then turn that into one of three actions. Everything downstream depends only
on `RelevanceScorer`, so the pipeline can be developed and tested against a stub on a
laptop and run against the fine-tuned Flan-T5 wherever the weights happen to live.

`T5Scorer` imports torch lazily: importing this module must stay free on the Mac.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from core.cache import get_cache

YES, NO = "yes", "no"
PROMPT = "Is this passage relevant to judging the claim?\nClaim: {claim}\nPassage: {passage}"


class RelevanceScorer(Protocol):
    """Score how relevant each passage is to the claim, in [-1, 1]."""

    def score(self, claim: str, passages: Sequence[str]) -> list[float]: ...


class LexicalScorer:
    """A stub that ranks by word overlap. For tests and for wiring work off-GPU.

    It is deliberately crude and deliberately *not* a fallback the pipeline reaches for
    on its own: a run that silently swapped the trained evaluator for word counting
    would produce plausible numbers that mean nothing. Callers ask for it by name.
    """

    _WORD = re.compile(r"[a-z0-9']+")

    def _terms(self, text: str) -> set[str]:
        return {w for w in self._WORD.findall(text.lower()) if len(w) > 3}

    def score(self, claim: str, passages: Sequence[str]) -> list[float]:
        want = self._terms(claim)
        if not want:
            return [0.0] * len(passages)
        out = []
        for passage in passages:
            overlap = len(want & self._terms(passage)) / len(want)
            # squash into [-1, 1] with 0 at ~25% overlap, which is roughly where the
            # trained model's decision boundary sits on this data
            out.append(max(-1.0, min(1.0, math.tanh(4 * (overlap - 0.25)))))
        return out


class T5Scorer:
    """The fine-tuned Flan-T5 evaluator from P1.3.

    The score is P("yes") against P("no") at the first decoded position, mapped to
    [-1, 1] -- the same quantity the training script reports AUC over, so dev numbers
    and inference-time behaviour cannot drift apart.
    """

    def __init__(self, model_dir: Path | str, batch_size: int = 16, max_tokens: int = 512):
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None
        self._yes = self._no = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

        from core.device import get_device  # noqa: PLC0415

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"No evaluator at {self.model_dir}. Train one with evaluator.train, or "
                "pass LexicalScorer() explicitly if you are only wiring things up."
            )
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir))
        self._model.to(get_device()).eval()
        self._yes = self._tokenizer(YES, add_special_tokens=False)["input_ids"][0]
        self._no = self._tokenizer(NO, add_special_tokens=False)["input_ids"][0]
        self._torch = torch

    def score(self, claim: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        self._load()
        torch = self._torch
        device = next(self._model.parameters()).device
        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(passages), self.batch_size):
                batch = passages[i : i + self.batch_size]
                enc = self._tokenizer(
                    [PROMPT.format(claim=claim, passage=p) for p in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_tokens,
                ).to(device)
                start = torch.full(
                    (len(batch), 1), self._model.config.decoder_start_token_id, device=device
                )
                logits = self._model(**enc, decoder_input_ids=start).logits[:, 0, :]
                probs = torch.softmax(logits[:, [self._yes, self._no]].float(), dim=-1)[:, 0]
                out += (probs * 2 - 1).cpu().tolist()
        return out


class CachedScorer:
    """Memoises an inner scorer (rule 1). Scoring the same passage twice is a bug."""

    def __init__(self, inner: RelevanceScorer, namespace: str = "evaluator"):
        self.inner = inner
        self.cache = get_cache(namespace)
        self.tag = type(inner).__name__

    def score(self, claim: str, passages: Sequence[str]) -> list[float]:
        scores: list[float | None] = []
        missing: list[tuple[int, str]] = []
        for i, passage in enumerate(passages):
            hit = self.cache.get({"scorer": self.tag, "claim": claim, "passage": passage})
            scores.append(hit)
            if hit is None:
                missing.append((i, passage))
        if missing:
            fresh = self.inner.score(claim, [p for _, p in missing])
            for (i, passage), value in zip(missing, fresh, strict=True):
                self.cache.set({"scorer": self.tag, "claim": claim, "passage": passage}, value)
                scores[i] = value
        return [float(s) for s in scores]


def load_scorer(model_dir: Path | str | None, cached: bool = True) -> RelevanceScorer:
    """`None` means the stub, chosen deliberately rather than fallen back to."""
    scorer: RelevanceScorer = LexicalScorer() if model_dir is None else T5Scorer(model_dir)
    return CachedScorer(scorer) if cached else scorer
