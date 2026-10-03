from pathlib import Path

import pytest

from core.config import RunConfig
from experience.format import Evidence, Trajectory, estimate_tokens, read_jsonl
from training.data import build_dataset, build_example, write_dataset
from training.neighbors import LexicalRetriever, is_self
from training.sft import parse_args, prepare

FIXTURE = Path(__file__).resolve().parent.parent / "training/fixtures/sample_trajectories.jsonl"


def traj(claim: str, verdict: str = "Supported", **kw) -> Trajectory:
    kw.setdefault("evidence", [Evidence(url="https://a.org/1", text="Some supporting text.")])
    return Trajectory(claim=claim, verdict=verdict, **kw)


# --- example construction ----------------------------------------------------------


def test_plain_example_has_the_three_roles():
    ex = build_example(traj("The sky is blue"))
    assert [m["role"] for m in ex.prompt] == ["system", "user"]
    assert ex.completion[0]["role"] == "assistant"


def test_completion_is_the_verdict():
    ex = build_example(traj("x", verdict="Refuted"))
    assert ex.completion[0]["content"].startswith("Verdict: Refuted")


def test_claim_and_evidence_reach_the_user_turn():
    ex = build_example(traj("Kochi metro opened in 2017"))
    user = ex.prompt[1]["content"]
    assert "Kochi metro opened in 2017" in user and "Some supporting text." in user


def test_retrieved_trajectories_appear_in_context():
    past = traj("An earlier related claim about metros")
    ex = build_example(traj("A new claim"), retrieved=[past])
    assert "Past verifications" in ex.prompt[1]["content"]
    assert "An earlier related claim" in ex.prompt[1]["content"]
    assert ex.n_retrieved == 1


def test_no_context_block_when_nothing_retrieved():
    assert "Past verifications" not in build_example(traj("A claim")).prompt[1]["content"]


def test_example_fits_max_seq_length():
    """A huge trajectory plus two huge neighbours must still fit the 8 GB budget."""
    big = [Evidence(url="https://a.org/1", text="word " * 20000) for _ in range(5)]
    neighbours = [traj(f"past {i}", evidence=big) for i in range(2)]
    ex = build_example(traj("current claim", evidence=big), retrieved=neighbours)
    total = sum(estimate_tokens(m["content"]) for m in ex.prompt + ex.completion)
    assert total <= RunConfig().max_seq_length


def test_oversized_context_drops_neighbours_before_evidence():
    cfg = RunConfig(max_seq_length=1200)  # too tight for 2 x 1000-token neighbours
    neighbours = [
        traj(f"past {i}", evidence=[Evidence(url="https://a.org/1", text="w " * 2000)])
        for i in range(2)
    ]
    ex = build_example(
        traj("current", evidence=[Evidence(url="https://b.org/1", text="keep me")]),
        retrieved=neighbours,
        cfg=cfg,
    )
    assert ex.n_retrieved < 2
    assert "keep me" in ex.prompt[1]["content"]
    assert ex.trimmed


def test_answer_is_never_truncated():
    long_rationale = "because " * 400
    ex = build_example(traj("c", rationale=long_rationale), cfg=RunConfig(max_seq_length=512))
    assert long_rationale in ex.completion[0]["content"]


# --- self-exclusion (rule 5) -------------------------------------------------------


def test_is_self_by_id():
    t = traj("a")
    assert is_self(t, t)


def test_is_self_by_claim_id():
    a = traj("phrasing one", claim_id="c-1")
    b = traj("phrasing two", claim_id="c-1")
    assert is_self(a, b)


def test_is_self_by_claim_text():
    assert is_self(traj("Same Claim"), traj("  same claim "))


def test_different_claims_are_not_self():
    assert not is_self(traj("a", claim_id="c-1"), traj("b", claim_id="c-2"))


def test_retriever_never_returns_the_query():
    pool = [traj(f"vaccine policy change in region {i}", claim_id=f"c-{i}") for i in range(10)]
    r = LexicalRetriever(pool)
    for t in pool:
        assert all(not is_self(t, n) for n in r.retrieve(t, k=2))


def test_retriever_prefers_lexical_overlap():
    pool = [
        traj("The Kochi metro opened in 2017", claim_id="c-1"),
        traj("The Kochi metro extension opened in 2019", claim_id="c-2"),
        traj("Penguins are flightless birds", claim_id="c-3"),
    ]
    best = LexicalRetriever(pool).retrieve(pool[0], k=1)
    assert best and "Kochi metro extension" in best[0].claim


def test_retriever_returns_nothing_from_a_single_item_pool():
    only = traj("alone", claim_id="c-1")
    assert LexicalRetriever([only]).retrieve(only, k=2) == []


# --- dataset assembly --------------------------------------------------------------


def test_incorrect_verifications_are_skipped_by_default():
    trajs = [
        traj("a", gold_label="Supported"),  # correct
        traj("b", verdict="Refuted", gold_label="Supported"),  # incorrect
    ]
    examples, stats = build_dataset(trajs)
    assert stats.n_examples == 1 and stats.n_skipped == 1


def test_keep_incorrect_trains_on_failures_too():
    trajs = [traj("b", verdict="Refuted", gold_label="Supported")]
    _, stats = build_dataset(trajs, keep_incorrect=True)
    assert stats.n_examples == 1 and stats.n_skipped == 0


def test_unlabelled_trajectories_are_kept():
    _, stats = build_dataset([traj("no gold label here")])
    assert stats.n_examples == 1


def test_plain_mode_retrieves_nothing():
    pool = [traj(f"claim about topic {i}", claim_id=f"c-{i}") for i in range(5)]
    _, stats = build_dataset(pool, mode="plain")
    assert stats.n_with_retrieval == 0


def test_retrieval_mode_adds_context():
    pool = [traj(f"vaccine policy change number {i}", claim_id=f"c-{i}") for i in range(5)]
    _, stats = build_dataset(pool, mode="retrieval")
    assert stats.n_with_retrieval > 0


def test_retrieval_mode_respects_k():
    pool = [traj(f"metro line {i} opened", claim_id=f"c-{i}") for i in range(8)]
    cfg = RunConfig(sft_k_trajectories=2)
    examples, _ = build_dataset(pool, mode="retrieval", cfg=cfg)
    assert all(e.n_retrieved <= 2 for e in examples)


def test_stats_count_verdicts():
    trajs = [traj("a"), traj("b", verdict="Refuted"), traj("c", verdict="Refuted")]
    _, stats = build_dataset(trajs)
    assert stats.verdict_counts == {"Supported": 1, "Refuted": 2}


def test_write_dataset_is_valid_jsonl(tmp_path):
    import json

    examples, _ = build_dataset([traj("a"), traj("b")])
    path = write_dataset(examples, tmp_path / "d.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2 and set(rows[0]) == {"prompt", "completion"}


# --- the fixture and the CLI -------------------------------------------------------


def test_fixture_is_readable():
    trajs = read_jsonl(FIXTURE)
    assert len(trajs) == 60
    assert all(t.evidence for t in trajs)


def test_fixture_contains_some_failures_to_skip():
    trajs = read_jsonl(FIXTURE)
    assert any(t.correct is False for t in trajs)


def test_smoke_flag_sets_the_small_run():
    args = parse_args(["--data", str(FIXTURE), "--name", "smoke", "--smoke"])
    assert args.limit == 50 and args.max_steps == 10 and args.save_steps == 5


def test_mode_must_be_known():
    with pytest.raises(SystemExit):
        parse_args(["--data", str(FIXTURE), "--name", "x", "--mode", "telepathy"])


def test_prepare_builds_a_dataset_without_torch():
    """The whole data path must run on the Mac -- this test proves it."""
    args = parse_args(
        ["--data", str(FIXTURE), "--name", "t", "--mode", "retrieval", "--limit", "20"]
    )
    records, stats, cfg = prepare(args)
    assert len(records) == stats["n_examples"] > 0
    assert cfg.experience_retrieval is True
    assert stats["max_tokens"] <= cfg.max_seq_length


def test_prepare_is_deterministic_for_a_seed():
    argv = ["--data", str(FIXTURE), "--name", "t", "--limit", "10", "--seed", "7"]
    first, _, _ = prepare(parse_args(argv))
    second, _, _ = prepare(parse_args(argv))
    assert first == second
