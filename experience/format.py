"""The one trajectory format (engineering rule 3).

A trajectory is one complete verification: the claim, the evidence that was retrieved
and how it was graded, the actions the agent took, and the verdict it reached. The same
objects are used for the FAISS store, for retrieval-augmented SFT, and for the poisoning
study, so that a trajectory written by the v0 agent trains the verifier unchanged.

The format freezes on Oct 8. After that, any change needs a logged decision and
invalidates every stored trajectory.

Everything here is pure Python: it imports no torch and no transformers, so it runs on
the Mac and in tests.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

VERDICTS: tuple[str, ...] = (
    "Supported",
    "Refuted",
    "Conflicting evidence",
    "Not enough evidence",
)

# CRAG's three-way route, recorded so the action share can be reported per run.
ACTIONS: tuple[str, ...] = ("correct", "incorrect", "ambiguous")

# Rough English estimate, used when no real tokenizer is available (i.e. on the Mac).
# Deliberately pessimistic: a cap enforced with this must also hold under a real
# tokenizer, or an example silently overflows `max_seq_length` on the GPU box.
_CHARS_PER_TOKEN = 3.5

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def _truncate_to_tokens(text: str, limit: int, count: TokenCounter) -> str:
    """Cut `text` down to `limit` tokens on a word boundary."""
    if limit <= 0 or count(text) <= limit:
        return text if count(text) <= limit else ""
    words = text.split()
    low, high = 0, len(words)
    while low < high:
        mid = (low + high + 1) // 2
        if count(" ".join(words[:mid])) <= limit:
            low = mid
        else:
            high = mid - 1
    return " ".join(words[:low])


@dataclass
class Evidence:
    """One retrieved passage, with where it came from and how it was graded."""

    url: str
    domain: str = ""
    title: str = ""
    text: str = ""
    grade: float | None = None  # evaluator relevance score, roughly [-1, 1]
    retrieved_by: str = "knowledge_store"  # knowledge_store | web_search
    poisoned: bool = False  # C1 bookkeeping only; never rendered into a prompt

    def __post_init__(self) -> None:
        if not self.domain and self.url:
            from retrieval.blocklist import domain_of

            self.domain = domain_of(self.url)

    def render(self, token_cap: int | None = None, count: TokenCounter = estimate_tokens) -> str:
        head = f"[{self.domain or 'unknown'}]"
        if self.title:
            head += f" {self.title}"
        body = self.text
        if token_cap is not None:
            body = _truncate_to_tokens(body, max(token_cap - count(head), 0), count)
        return f"{head}\n{body}".strip()


@dataclass
class Step:
    """One action in the trace, so a run can be read back and audited."""

    action: str  # retrieve | grade | route | rewrite | search | refine | verify
    detail: str = ""
    score: float | None = None


@dataclass
class Trajectory:
    claim: str
    verdict: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    rationale: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)

    # Provenance and evaluation
    claim_id: str | None = None  # dataset id, for self-exclusion (rule 5)
    gold_label: str | None = None
    action: str | None = None  # CRAG route taken
    source: str = "v0_agent"  # v0_agent | crag | poisoned | human
    dataset: str = "averitec"
    split: str = "train"
    seed: int | None = None

    # C1 bookkeeping
    poisoned: bool = False

    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: date.today().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict {self.verdict!r} not one of {VERDICTS}")
        if self.action is not None and self.action not in ACTIONS:
            raise ValueError(f"action {self.action!r} not one of {ACTIONS}")

    @property
    def correct(self) -> bool | None:
        """None when the claim has no gold label, so unlabelled runs stay usable."""
        if self.gold_label is None:
            return None
        return self.verdict == self.gold_label

    @property
    def domains(self) -> list[str]:
        """Distinct source domains, in first-seen order. Feeds the C3 trust update."""
        seen: dict[str, None] = {}
        for e in self.evidence:
            if e.domain:
                seen.setdefault(e.domain, None)
        return list(seen)

    # --- rendering -----------------------------------------------------------------

    def render_evidence(
        self,
        token_cap: int | None = None,
        count: TokenCounter = estimate_tokens,
        max_items: int | None = None,
    ) -> str:
        items = self.evidence[:max_items] if max_items else self.evidence
        if not items:
            return "(no evidence retrieved)"
        per_item = token_cap // len(items) if token_cap else None
        return "\n\n".join(
            f"{i}. {e.render(per_item, count)}" for i, e in enumerate(items, start=1)
        )

    def render_answer(self) -> str:
        """The assistant target for SFT: verdict first so it is trivially parseable."""
        out = f"Verdict: {self.verdict}"
        return f"{out}\nReasoning: {self.rationale}" if self.rationale else out

    def render_for_context(
        self,
        token_cap: int = 1000,
        count: TokenCounter = estimate_tokens,
    ) -> str:
        """Compact form used when this trajectory is retrieved into another's context.

        Capped at `token_cap` (default 1,000) so that k of these plus the
        current claim's evidence stay inside `max_seq_length` on an 8 GB GPU.
        """
        header = f"Past claim: {self.claim}\n{self.render_answer()}"
        remaining = token_cap - count(header) - 12  # headroom for the evidence label
        if remaining <= 0:
            return _truncate_to_tokens(header, token_cap, count)
        return f"{header}\nEvidence used:\n{self.render_evidence(remaining, count)}"

    # --- serialisation -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        data = dict(data)
        data.pop("correct", None)  # derived property, ignore if a writer included it
        evidence = [Evidence(**e) for e in data.pop("evidence", [])]
        steps = [Step(**s) for s in data.pop("steps", [])]
        known = {f.name for f in cls.__dataclass_fields__.values()}
        extra = {k: v for k, v in data.items() if k not in known}
        kwargs = {k: v for k, v in data.items() if k in known}
        traj = cls(evidence=evidence, steps=steps, **kwargs)
        traj.metadata.update(extra)
        return traj


def write_jsonl(trajectories: Iterable[Trajectory], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for t in trajectories:
            fh.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path | str) -> list[Trajectory]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path | str) -> Iterator[Trajectory]:
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Trajectory.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: bad trajectory - {exc}") from exc
