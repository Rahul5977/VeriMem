"""Run configuration. Every experiment writes its config next to its metrics (rule 7).

With all new flags off, results must reproduce the Phase 1-2 numbers (rule 4), so
`RunConfig()` with no arguments is the plain-CRAG baseline, not the full system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

Verifier = Literal["prompted", "finetuned"]


@dataclass
class RunConfig:
    # --- contribution flags (all off = Phase 1-2 baseline) ---
    experience_retrieval: bool = False  # ExpRAG: retrieve past trajectories
    gate_experience: bool = False  # C2: evaluator-gated experience retrieval
    source_trust: bool = False  # C3: Bayesian per-domain trust
    verifier: Verifier = "prompted"

    # --- CRAG ---
    crag_upper_threshold: float = 0.6  # above -> Correct (refine)
    crag_lower_threshold: float = -0.6  # below -> Incorrect (rewrite + web search)
    top_k_evidence: int = 5
    top_k_strips: int = 5

    # --- experience store ---
    k_trajectories: int = 3
    gate_threshold: float = 0.0  # C2: drop trajectories scoring below this
    trust_floor: float = 0.3  # C3: drop sources whose trust falls below this

    # --- models ---
    # Teacher + judge: one hosted Qwen model for the whole project (rule 8). The ID is
    # fixed in P0.13 after a 20-claim comparison; "TBD" until then.
    api_model: str = "TBD"
    api_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    disable_thinking: bool = True  # rule 8: judge calls run with thinking off
    evaluator_model: str = "google/flan-t5-base"
    verifier_model: str = "Qwen/Qwen2.5-3B-Instruct"  # the student; never the teacher
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- training memory budget (GPU-A has 8 GB of VRAM) ---
    sft_k_trajectories: int = 2  # k <= 2 in the retrieval-augmented SFT context
    trajectory_token_cap: int = 1000  # per retrieved trajectory
    max_seq_length: int = 4096  # a training example must stay under this
    grad_accum_steps: int = 16  # with batch size 1
    checkpoint_every_steps: int = 200  # every run must be resumable

    # --- data / reproducibility ---
    dataset: str = "averitec"
    split: str = "dev"
    seed: int = 13
    limit: int | None = None  # cap claims, for pilots

    # --- attacks ---
    memory_poison_rate: float = 0.0
    source_poison: bool = False

    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, directory: Path) -> Path:
        import yaml

        path = Path(directory) / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        import yaml

        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        leftover = {k: v for k, v in data.items() if k not in known}
        cfg = cls(**kwargs)
        cfg.extra.update(leftover)
        return cfg

    def with_(self, **overrides: Any) -> RunConfig:
        return RunConfig.from_dict({**self.to_dict(), **overrides})


BASELINE = RunConfig()
FULL_VERIMEM = RunConfig(
    experience_retrieval=True,
    gate_experience=True,
    source_trust=True,
    verifier="finetuned",
)
