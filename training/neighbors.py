"""Neighbour lookup for building retrieval-augmented SFT data.

P2.6 builds the real FAISS store over sentence-transformer embeddings. Until then this
lexical retriever lets the training-data pipeline be written, tested and smoke-trained
without torch, so it runs on the Mac. Swap in the FAISS store through the same
`Retriever` signature once it exists -- nothing downstream changes.

Rule 5: a trajectory is never retrieved into its own training example. Exclusion is by
id, by dataset claim id, and by exact claim text, because the same claim can appear
under more than one trajectory id (e.g. a re-run, or a poisoned twin).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

from experience.format import Trajectory

Retriever = Callable[[Trajectory, int], list[Trajectory]]

_WORD = re.compile(r"[a-z0-9']+")

# Common words carry no retrieval signal and swamp the overlap score on short claims.
_STOPWORDS = frozenset(
    """a an and are as at be been by for from has have in is it its of on or that the
    to was were will with this these those there their they he she his her not no""".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


class LexicalRetriever:
    """BM25-lite over claim text. A stand-in for the FAISS store (P2.6)."""

    def __init__(self, pool: Sequence[Trajectory], k1: float = 1.5, b: float = 0.75) -> None:
        self.pool = list(pool)
        self.k1, self.b = k1, b
        self._docs = [Counter(tokenize(t.claim)) for t in self.pool]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(d.keys())
        n = len(self._docs)
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def _score(self, query: Counter[str], index: int) -> float:
        doc, length = self._docs[index], self._lengths[index]
        if not length:
            return 0.0
        norm = self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1))
        return sum(
            self._idf.get(term, 0.0) * (doc[term] * (self.k1 + 1)) / (doc[term] + norm)
            for term in query
            if term in doc
        )

    def retrieve(self, query: Trajectory, k: int) -> list[Trajectory]:
        """Top-k neighbours of `query`, with `query` itself excluded (rule 5)."""
        if k <= 0:
            return []
        q = Counter(tokenize(query.claim))
        scored = [
            (self._score(q, i), i)
            for i, candidate in enumerate(self.pool)
            if not is_self(query, candidate)
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [self.pool[i] for score, i in scored[:k] if score > 0]


def is_self(query: Trajectory, candidate: Trajectory) -> bool:
    """True when `candidate` must be excluded from `query`'s retrieval (rule 5)."""
    if candidate.id == query.id:
        return True
    if query.claim_id is not None and candidate.claim_id == query.claim_id:
        return True
    return candidate.claim.strip().lower() == query.claim.strip().lower()
