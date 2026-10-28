"""The G2 table's four rows depend on two things that did not exist (P2.11, C4).

`run_baselines` never passed `memory=` to the agent, so the two retrieval rows were
unrunnable; and the student could only be served *with* an adapter, so the two prompted
rows had no path that matched the fine-tuned ones. These tests cover both, plus the
property the table rests on: all rows must see the same claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.datasets import sample
from eval.run_baselines import parse_args
from training.verifier import FinetunedVerifier, load_verifier


@pytest.fixture
def adapter(tmp_path: Path) -> Path:
    d = tmp_path / "run" / "adapter"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct"})
    )
    (d / "adapter_model.safetensors").write_bytes(b"x" * 32)
    return d


# --- serving the base model (the prompted rows) --------------------------------------


def test_the_student_can_be_served_with_no_adapter() -> None:
    """The prompted row. Not a degenerate case -- half the table needs it."""
    v = load_verifier(None, base_only=True)
    assert isinstance(v, FinetunedVerifier)
    assert v.adapter is None
    assert v.base_model == "Qwen/Qwen2.5-3B-Instruct"


def test_the_prompted_row_names_itself_as_prompted() -> None:
    """It lands in every trajectory's metadata; "no adapter" must be visible there."""
    assert "no adapter" in load_verifier(None, base_only=True).model


def test_base_and_adapter_are_mutually_exclusive(adapter: Path) -> None:
    """Passing both would silently serve one of them and report the other."""
    with pytest.raises(SystemExit, match="mutually exclusive"):
        load_verifier(adapter, base_only=True)


def test_the_base_rows_do_not_collide_in_the_cache(adapter: Path) -> None:
    """Base and adapter answers must not share a key, or one row reports the other's."""
    base = load_verifier(None, base_only=True)
    tuned = FinetunedVerifier(adapter)
    messages = [{"role": "user", "content": "Claim: x"}]
    assert base.cache_key(messages, 128) != tuned.cache_key(messages, 128)


def test_the_prompted_rows_share_a_cache_entry_with_each_other() -> None:
    """Two base-only verifiers are the same model, so they should not recompute."""
    a, b = load_verifier(None, base_only=True), load_verifier(None, base_only=True)
    messages = [{"role": "user", "content": "Claim: x"}]
    assert a.cache_key(messages, 128) == b.cache_key(messages, 128)


# --- the runner's new flags -----------------------------------------------------------


def _args(*extra: str):
    return parse_args(["--data", "data/averitec/dev.json", "--system", "crag", *extra])


def test_experience_retrieval_is_off_by_default() -> None:
    """Rule 4: with every new flag off, the run must reproduce the Phase 1 numbers."""
    args = _args()
    assert args.experience_retrieval is False
    assert args.gate_experience is False
    assert args.source_trust is False
    assert args.verifier_base is False
    assert args.verifier_adapter is None


def test_the_retrieval_flags_are_accepted_together() -> None:
    args = _args(
        "--trajectories", "t.jsonl", "--experience-retrieval", "--gate-experience", "--source-trust"
    )
    assert args.experience_retrieval and args.gate_experience and args.source_trust
    assert str(args.trajectories) == "t.jsonl"


def test_retrieval_without_trajectories_is_rejected() -> None:
    """Silently retrieving from an empty store would report ExpRAG as having no effect."""
    from eval import run_baselines

    with pytest.raises(SystemExit, match="needs --trajectories"):
        run_baselines.main(
            [
                "--data",
                "data/averitec/dev.json",
                "--system",
                "crag",
                "--experience-retrieval",
                "--dry-run",
            ]
        )


def test_retrieval_is_rejected_for_the_plain_baselines() -> None:
    """no_retrieval and plain_rag are *defined* without experience; silently adding it
    would make those rows something other than what the table calls them."""
    from eval import run_baselines

    with pytest.raises(SystemExit, match="only applies to the crag system"):
        run_baselines.main(
            [
                "--data",
                "data/averitec/dev.json",
                "--system",
                "plain_rag",
                "--trajectories",
                "t.jsonl",
                "--experience-retrieval",
                "--dry-run",
            ]
        )


# --- comparability across rows --------------------------------------------------------


def test_every_row_sees_the_same_claims() -> None:
    """Six rows are compared against each other. A different subset per row would make
    the differences between them meaningless."""
    from eval.datasets import Claim

    claims = [
        Claim(id=f"c{i}", claim=f"Claim number {i}.", gold_label="Refuted") for i in range(50)
    ]
    first = [c.id for c in sample(claims, 20, seed=13)]
    second = [c.id for c in sample(claims, 20, seed=13)]
    assert first == second
    # And a different seed really does pick differently, so the fixture above is not
    # passing for a trivial reason.
    assert [c.id for c in sample(claims, 20, seed=14)] != first
