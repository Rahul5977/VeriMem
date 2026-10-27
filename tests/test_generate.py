import json

import pytest

from experience.format import Evidence, Trajectory
from experience.generate import main as gen_main
from experience.generate import summarise


def traj(verdict="Supported", gold="Supported", action="correct", parsed=True, n_ev=2):
    return Trajectory(
        claim="a claim",
        verdict=verdict,
        gold_label=gold,
        action=action,
        evidence=[Evidence(url=f"https://a.org/{i}", text="t") for i in range(n_ev)],
        metadata={"parsed": parsed},
    )


# --- the summary --------------------------------------------------------------------


def test_counts_successes_and_failures():
    s = summarise([traj(), traj(verdict="Refuted"), traj()])
    assert s["total"] == 3 and s["correct"] == 2 and s["incorrect"] == 1


def test_unlabelled_are_counted_separately():
    s = summarise([traj(gold=None), traj()])
    assert s["unlabelled"] == 1


def test_action_share_sums_to_one():
    s = summarise([traj(action="correct"), traj(action="incorrect"), traj(action="ambiguous")])
    assert sum(s["action_share"].values()) == pytest.approx(1.0, abs=0.01)


def test_unparseable_is_surfaced():
    assert summarise([traj(parsed=False), traj()])["unparseable"] == 1


def test_macro_f1_is_reported_when_labels_exist():
    s = summarise([traj(), traj(verdict="Refuted", gold="Refuted")])
    assert s["macro_f1"] == 1.0


def test_no_macro_f1_without_labels():
    assert "macro_f1" not in summarise([traj(gold=None)])


def test_mean_evidence():
    assert summarise([traj(n_ev=2), traj(n_ev=4)])["mean_evidence"] == 3.0


def test_empty_input_is_safe():
    assert summarise([])["total"] == 0


# --- the CLI guards -----------------------------------------------------------------


def _files(tmp_path):
    claims = tmp_path / "train.json"
    claims.write_text(
        json.dumps(
            [
                {
                    "claim": "The Kochi metro opened in 2017.",
                    "label": "Supported",
                    "questions": [{"answers": [{"source_url": "https://a.org/1"}]}],
                }
            ]
        )
    )
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps(
            {"url": "https://a.org/1", "url2text": ["The Kochi metro opened in June 2017."] * 4}
        )
    )
    return claims, pool


def test_warns_when_generating_from_dev(tmp_path, monkeypatch, capsys):
    """Rule 5: trajectories become both training data and retrievable experience."""
    monkeypatch.setenv("VERIMEM_API_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("VERIMEM_API_MODEL", "qwen2.5:7b")
    claims, pool = _files(tmp_path)
    dev = tmp_path / "dev.json"
    dev.write_text(claims.read_text())
    gen_main(["--data", str(dev), "--store", str(pool), "--dry-run"])
    assert "own test set" in capsys.readouterr().out


def test_warns_without_a_trained_evaluator(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VERIMEM_API_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("VERIMEM_API_MODEL", "qwen2.5:7b")
    claims, pool = _files(tmp_path)
    gen_main(["--data", str(claims), "--store", str(pool), "--dry-run"])
    assert "lexical stub" in capsys.readouterr().out


def test_dry_run_reports_how_many_claims_have_evidence(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VERIMEM_API_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("VERIMEM_API_MODEL", "qwen2.5:7b")
    claims, pool = _files(tmp_path)
    gen_main(["--data", str(claims), "--store", str(pool), "--dry-run"])
    out = capsys.readouterr().out
    assert "store has documents for" in out and "nothing generated" in out
