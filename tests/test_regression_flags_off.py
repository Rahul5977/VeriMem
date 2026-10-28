"""Rule 4: with every new flag off, the pipeline must behave exactly as it did in Phase 1.

Phases 2 and 3 added experience retrieval, evaluator gating, Bayesian source trust and a
fine-tuned verifier. Each is meant to be inert until switched on, so that the Phase 1
baseline numbers remain the baseline and every later comparison is against something that
did not move. Nothing enforced that until this file.

**The test is not a stored golden string.** A golden file pins today's prompt and has to
be regenerated whenever anything legitimately changes, which trains everyone to
regenerate it without reading the diff. What rule 4 actually promises is a *relative*
property, so that is what is asserted: with the flags off, handing the pipeline a memory
store and a trust model must produce byte-identical calls to not handing it them at all.
That is the regression, stated directly and with nothing to maintain.

**Both halves are needed.** A flag wired to nothing would also satisfy the above, so each
flag is separately shown to change behaviour when it is on. A test that only proved
inertness would pass just as well on a pipeline where none of the contributions worked.

The numeric half -- that the flags-off run still scores what Phase 1 scored -- needs a
GPU session, so it reads `tests/fixtures/phase1_baseline.json` when that has been filled
in from a real run and skips otherwise. It never silently passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.crag import run_crag
from core.cache import CacheStats, DiskCache
from core.config import BASELINE, FULL_VERIMEM, RunConfig
from core.llm import LLMClient, SpendLedger
from eval.datasets import Claim
from evaluator.score import LexicalScorer
from experience.format import Evidence, Step, Trajectory
from experience.store import BetaTrust, ExperienceStore
from retrieval.store import PooledKnowledgeStore

OLLAMA = "http://localhost:11434/v1"

ROWS = [
    {
        "url": "https://reuters.com/a",
        "url2text": [
            "The Kochi Metro began commercial operations in June 2017 after several delays.",
            "The project was funded jointly by the state and central governments.",
            "Ridership in the first year fell short of the original projections.",
        ],
    },
    {
        "url": "https://bbc.co.uk/b",
        "url2text": [
            "Kochi's metro system was inaugurated in 2017 by the prime minister.",
            "It was the first metro in the state of Kerala.",
        ],
    },
]


def _llm(tmp_path: Path, reply: str = "Verdict: Supported\nReasoning: yes."):
    """An LLMClient that records every payload instead of calling anything."""
    sent: list[dict] = []

    def transport(payload):
        sent.append(payload)
        return {
            "model": "m",
            "choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }

    llm = LLMClient(base_url=OLLAMA, model="m", transport=transport)
    llm.cache = DiskCache("llm", root=tmp_path, stats=CacheStats())
    llm.ledger = SpendLedger(path=tmp_path, cap_usd=None)
    llm.sent = sent
    return llm


def _pool(tmp_path: Path) -> PooledKnowledgeStore:
    tmp_path.mkdir(parents=True, exist_ok=True)  # callers pass subdirectories
    p = tmp_path / "pool.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in ROWS), encoding="utf-8")
    return PooledKnowledgeStore(p)


def _claim() -> Claim:
    return Claim(
        id="c1", claim="The Kochi metro opened to the public in 2017.", gold_label="Supported"
    )


def _past(n: int = 6) -> list[Trajectory]:
    """A history that is *relevant* to the claim, so an inert flag cannot hide behind
    retrieval finding nothing."""
    return [
        Trajectory(
            claim=f"The Kochi metro opened to the public in 201{i}.",
            verdict="Refuted",
            rationale="The record does not support it.",
            evidence=[
                Evidence(url=f"https://example{i}.com/x", text="A passage about the Kochi metro.")
            ],
            steps=[Step("retrieve", "top-5")],
            claim_id=f"past-{i}",
            gold_label="Refuted",
        )
        for i in range(n)
    ]


def _machinery(tmp_path: Path, cfg: RunConfig):
    """Everything Phases 2-3 added, ready to be handed to the pipeline."""
    past = _past()
    trust = BetaTrust()
    for t in past:
        trust.observe(t)
    store = ExperienceStore(
        past,
        scorer=LexicalScorer(),
        trust=trust,
        gate_threshold=cfg.gate_threshold,
        trust_floor=cfg.trust_floor,
    )
    return store, trust


def _run(tmp_path: Path, cfg: RunConfig, *, with_machinery: bool):
    llm = _llm(tmp_path)
    kwargs = {}
    if with_machinery:
        store, trust = _machinery(tmp_path, cfg)
        kwargs = {"memory": store, "trust": trust}
    result = run_crag(_claim(), llm, _pool(tmp_path), LexicalScorer(), cfg, **kwargs)
    return result, llm.sent


# --- the baseline config is actually all-off ------------------------------------------


def test_the_default_config_has_every_new_flag_off() -> None:
    """If a default ever flips, every Phase 1 number silently stops being a baseline."""
    cfg = RunConfig()
    assert cfg.experience_retrieval is False
    assert cfg.gate_experience is False
    assert cfg.source_trust is False
    assert cfg.verifier == "prompted"
    assert cfg == BASELINE


def test_the_full_system_config_has_them_on() -> None:
    """The other end of rule 4: the shipped configuration is not accidentally the baseline."""
    assert FULL_VERIMEM.experience_retrieval is True
    assert FULL_VERIMEM.gate_experience is True
    assert FULL_VERIMEM.source_trust is True
    assert FULL_VERIMEM.verifier == "finetuned"


# --- inertness: the machinery changes nothing while switched off ----------------------


def test_flags_off_sends_byte_identical_calls_with_and_without_the_machinery(
    tmp_path: Path,
) -> None:
    """Rule 4, stated directly. This is the whole point of the file."""
    bare, bare_sent = _run(tmp_path / "a", RunConfig(), with_machinery=False)
    armed, armed_sent = _run(tmp_path / "b", RunConfig(), with_machinery=True)

    assert len(bare_sent) == len(armed_sent), "a different number of model calls was made"
    for i, (x, y) in enumerate(zip(bare_sent, armed_sent, strict=True)):
        assert x == y, f"call {i} differs when the inert machinery is present"
    assert bare.verdict == armed.verdict


def test_flags_off_retrieves_no_experience_even_from_a_relevant_store(
    tmp_path: Path,
) -> None:
    result, _ = _run(tmp_path, RunConfig(), with_machinery=True)
    assert result.n_memory == 0
    assert result.retrieved_ids == []


def test_flags_off_never_consults_the_trust_model(tmp_path: Path) -> None:
    """None, not an all-zeros dict. A zeros dict is truthy and leaked a `trust` key into
    every flags-off trajectory's metadata, which is what this file caught."""
    result, _ = _run(tmp_path, RunConfig(), with_machinery=True)
    assert result.trust_stats is None


def test_a_flags_off_trajectory_carries_no_phase3_metadata(tmp_path: Path) -> None:
    """A Phase 1 trajectory and a flags-off Phase 3 trajectory must be comparable records,
    not differ by keys that only exist now."""
    from agent.crag import crag

    llm = _llm(tmp_path)
    store, trust = _machinery(tmp_path, RunConfig())
    t = crag(
        _claim(),
        llm,
        _pool(tmp_path),
        LexicalScorer(),
        RunConfig(),
        memory=store,
        trust=trust,
    )
    assert "trust" not in t.metadata
    assert "retrieved_ids" not in t.metadata
    assert t.metadata["verifier"] == "prompted"


def test_flags_off_uses_the_teacher_as_the_verifier(tmp_path: Path) -> None:
    """No adapter given means the teacher answers -- not a silent fall back to a student."""
    llm = _llm(tmp_path)
    result = run_crag(_claim(), llm, _pool(tmp_path), LexicalScorer(), RunConfig())
    assert result.verdict == "Supported"
    assert llm.sent, "the teacher was never called"


# --- non-vacuity: each flag must actually do something when on ------------------------


def test_experience_retrieval_changes_the_prompt_when_on(tmp_path: Path) -> None:
    """Without this, a flag wired to nothing would pass the inertness tests above."""
    off, off_sent = _run(tmp_path / "off", RunConfig(), with_machinery=True)
    on, on_sent = _run(tmp_path / "on", RunConfig(experience_retrieval=True), with_machinery=True)
    assert on.n_memory > 0, "the store had relevant history and none was retrieved"
    assert on_sent != off_sent
    assert any("Past verifications" in str(c.get("messages")) for c in on_sent)
    assert not any("Past verifications" in str(c.get("messages")) for c in off_sent)


def test_gating_changes_what_survives_retrieval(tmp_path: Path) -> None:
    plain, _ = _run(tmp_path / "p", RunConfig(experience_retrieval=True), with_machinery=True)
    gated, _ = _run(
        tmp_path / "g",
        RunConfig(experience_retrieval=True, gate_experience=True, gate_threshold=0.99),
        with_machinery=True,
    )
    assert plain.n_memory > 0
    assert gated.n_memory < plain.n_memory, "an impossible gate threshold kept everything"


def test_source_trust_changes_the_evidence_when_on(tmp_path: Path) -> None:
    off, _ = _run(tmp_path / "off", RunConfig(), with_machinery=True)
    on, _ = _run(
        tmp_path / "on", RunConfig(source_trust=True, trust_floor=0.99), with_machinery=True
    )
    assert off.trust_stats is None
    assert on.trust_stats is not None
    assert on.trust_stats["strips_considered"] > 0
    assert on.trust_stats["strips_dropped_by_trust"] > 0


# --- the numeric half -----------------------------------------------------------------

BASELINE_FILE = Path(__file__).parent / "fixtures" / "phase1_baseline.json"


def test_the_flags_off_numbers_still_match_phase_1() -> None:
    """Reads a baseline recorded from a real run; skips loudly rather than passing when
    it has not been filled in yet."""
    if not BASELINE_FILE.exists():
        pytest.skip(f"no recorded baseline at {BASELINE_FILE.name}")
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    if not baseline.get("systems"):
        pytest.skip("baseline file is a placeholder; fill it from a G1 run")

    results = sorted(Path("results").glob("*_g1-rerun/metrics.json"))
    if not results:
        pytest.skip("no g1-rerun results on this machine")

    metrics = json.loads(results[-1].read_text(encoding="utf-8"))
    tol = baseline.get("tolerance", 0.005)
    for system, expected in baseline["systems"].items():
        got = metrics.get(system, {}).get("macro_f1")
        assert got is not None, f"{system} is missing from {results[-1]}"
        assert abs(got - expected) <= tol, (
            f"{system} macro-F1 moved from {expected} to {got} (tolerance {tol}). "
            "Either a change altered the flags-off pipeline, or the baseline needs "
            "re-recording with a stated reason."
        )
