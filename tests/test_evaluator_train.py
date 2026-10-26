import json

import numpy as np
import pytest

from evaluator.train import (
    PROMPT,
    _relevance_metrics,
    class_balance,
    describe,
    format_example,
    load_pairs,
    parse_args,
)


def write_split(tmp_path, train_rows, dev_rows):
    d = tmp_path / "evaluator"
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in (("relevance_train.jsonl", train_rows), ("relevance_dev.jsonl", dev_rows)):
        (d / name).write_text("\n".join(json.dumps(r) for r in rows))
    return d


def pair(claim_id, label=1, kind="positive"):
    return {
        "claim": f"claim {claim_id}",
        "passage": f"passage for {claim_id}",
        "url": "https://a.org/1",
        "label": label,
        "kind": kind,
        "claim_id": claim_id,
        "split": "train",
    }


# --- formatting ---------------------------------------------------------------------


def test_prompt_carries_claim_and_passage():
    ex = format_example(pair("c1"))
    assert "claim c1" in ex["input"] and "passage for c1" in ex["input"]


def test_target_is_yes_for_relevant_and_no_otherwise():
    assert format_example(pair("c1", label=1))["target"] == "yes"
    assert format_example(pair("c1", label=0))["target"] == "no"


def test_prompt_template_is_a_question():
    assert PROMPT.startswith("Is this passage relevant")


# --- loading and balance ------------------------------------------------------------


def test_load_respects_the_limit(tmp_path):
    d = write_split(tmp_path, [pair(f"c{i}") for i in range(10)], [pair("d1")])
    assert len(load_pairs(d / "relevance_train.jsonl", limit=3)) == 3


def test_class_balance_counts_kinds():
    rows = [pair("a"), pair("b", 0, "hard_negative"), pair("c", 0, "easy_negative")]
    b = class_balance(rows)
    assert b["n"] == 3 and b["positive"] == 1 and b["negative"] == 2
    assert b["by_kind"]["hard_negative"] == 1


# --- the leakage guard --------------------------------------------------------------


def test_describe_rejects_a_claim_in_both_splits(tmp_path):
    """Rule 5: the split must be by claim, or dev accuracy is meaningless."""
    d = write_split(tmp_path, [pair("shared")], [pair("shared")])
    with pytest.raises(SystemExit, match="both train and dev"):
        describe(parse_args(["--data", str(d)]))


def test_describe_accepts_a_clean_split(tmp_path):
    d = write_split(tmp_path, [pair("a")], [pair("b")])
    stats = describe(parse_args(["--data", str(d)]))
    assert stats["train"]["n"] == 1 and stats["dev"]["n"] == 1


def test_describe_refuses_an_empty_dataset(tmp_path):
    d = write_split(tmp_path, [], [])
    with pytest.raises(SystemExit, match="No training pairs"):
        describe(parse_args(["--data", str(d)]))


# --- metrics ------------------------------------------------------------------------


def test_perfect_separation_scores_auc_one():
    m = _relevance_metrics(np.array([0.9, 0.8, -0.7, -0.9]), np.array([1, 1, 0, 0]))
    assert m["auc"] == 1.0 and m["accuracy"] == 1.0


def test_inverted_scores_score_auc_zero():
    m = _relevance_metrics(np.array([-0.9, -0.8, 0.7, 0.9]), np.array([1, 1, 0, 0]))
    assert m["auc"] == 0.0


def test_random_scores_sit_near_half():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=400)
    labels = rng.integers(0, 2, size=400)
    assert 0.4 < _relevance_metrics(scores, labels)["auc"] < 0.6


def test_ties_count_as_half():
    m = _relevance_metrics(np.array([0.5, 0.5]), np.array([1, 0]))
    assert m["auc"] == 0.5


def test_auc_is_nan_when_a_class_is_missing():
    import math

    m = _relevance_metrics(np.array([0.5, 0.6]), np.array([1, 1]))
    assert math.isnan(m["auc"])


def test_metrics_report_the_mean_score_per_class():
    m = _relevance_metrics(np.array([0.8, -0.6]), np.array([1, 0]))
    assert m["mean_score_positive"] == 0.8 and m["mean_score_negative"] == -0.6


# --- CLI ----------------------------------------------------------------------------


def test_smoke_flag_shrinks_the_run():
    args = parse_args(["--smoke"])
    assert args.limit == 200 and args.epochs == 1.0


def test_dry_run_needs_no_gpu(tmp_path, capsys):
    from evaluator.train import main

    d = write_split(tmp_path, [pair(f"c{i}") for i in range(4)], [pair("d1")])
    assert main(["--data", str(d), "--dry-run"]) == 0
    assert "nothing trained" in capsys.readouterr().out
