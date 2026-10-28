import json

from eval.run_poisoning import CONFIGS, RATES, build_trust
from eval.run_poisoning import main as poison_main
from experience.format import Evidence, Trajectory


def traj(claim, domains=("good.com",), correct=True):
    return Trajectory(
        claim=claim,
        verdict="Supported",
        gold_label="Supported" if correct else "Refuted",
        evidence=[Evidence(url=f"https://{d}/a", text="t") for d in domains],
    )


# --- the configurations under test --------------------------------------------------


def test_three_configurations_cover_the_hypotheses():
    assert CONFIGS["no_memory"] == (False, False, False)
    assert CONFIGS["memory"] == (True, False, False)
    assert CONFIGS["gated_memory"] == (True, True, True)


def test_rates_span_the_planned_sweep():
    assert RATES == (0.0, 0.05, 0.10, 0.20)


def test_no_memory_is_the_control():
    """The adversary has nothing to write to, so this arm must not vary with rate.
    If it does, the harness is leaking, not the system."""
    assert CONFIGS["no_memory"][0] is False


# --- trust is fitted on clean history -----------------------------------------------


def test_trust_learns_from_clean_history():
    t = build_trust([traj("a", ("bbc.co.uk",), correct=True) for _ in range(5)])
    assert t.trust("bbc.co.uk") > 0.5


def test_trust_never_sees_the_attack_domains():
    """Fitting trust on the poisoned store would hand C3 the answer: it would learn the
    attack domains are bad from the attack itself, which no deployed system could do
    before being attacked."""
    t = build_trust([traj("a", ("bbc.co.uk",))])
    assert t.trust("americasfreedomfighters.com") == 0.5  # unseen, so the prior


def test_trust_blames_domains_behind_wrong_verdicts():
    t = build_trust([traj("a", ("bad.com",), correct=False) for _ in range(5)])
    assert t.trust("bad.com") < 0.5


# --- the CLI ------------------------------------------------------------------------


def _files(tmp_path):
    claims = tmp_path / "dev.json"
    claims.write_text(
        json.dumps(
            [
                {
                    "claim": f"Claim {i} about a policy decision taken in 2020.",
                    "label": [
                        "Supported",
                        "Refuted",
                        "Not Enough Evidence",
                        "Conflicting Evidence/Cherrypicking",
                    ][i % 4],
                    "questions": [{"answers": [{"source_url": "https://a.org/1"}]}],
                }
                for i in range(12)
            ]
        )
    )
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps(
            {"url": "https://a.org/1", "url2text": ["A policy decision was taken in 2020."] * 4}
        )
    )
    trajectories = tmp_path / "traj.jsonl"
    trajectories.write_text(
        "\n".join(json.dumps(traj(f"Past claim {i} about policy").to_dict()) for i in range(20))
    )
    return claims, pool, trajectories


def test_dry_run_reports_the_grid_without_running(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VERIMEM_API_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("VERIMEM_API_MODEL", "qwen2.5:7b")
    claims, pool, trajectories = _files(tmp_path)
    poison_main(
        [
            "--data",
            str(claims),
            "--store",
            str(pool),
            "--trajectories",
            str(trajectories),
            "--targets",
            "4",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert "targets" in out and "cells" in out and "nothing run" in out


def test_dry_run_warns_without_an_evaluator(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VERIMEM_API_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("VERIMEM_API_MODEL", "qwen2.5:7b")
    claims, pool, trajectories = _files(tmp_path)
    poison_main(
        [
            "--data",
            str(claims),
            "--store",
            str(pool),
            "--trajectories",
            str(trajectories),
            "--targets",
            "4",
            "--dry-run",
        ]
    )
    assert "H2 would be testing word counting" in capsys.readouterr().out


def test_every_target_is_evaluated(tmp_path, monkeypatch, capsys):
    """A target whose claim is never run cannot have its attack observed."""
    monkeypatch.setenv("VERIMEM_API_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("VERIMEM_API_MODEL", "qwen2.5:7b")
    claims, pool, trajectories = _files(tmp_path)
    poison_main(
        [
            "--data",
            str(claims),
            "--store",
            str(pool),
            "--trajectories",
            str(trajectories),
            "--targets",
            "8",
            "--limit",
            "2",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    n_targets = int(out.split("targets,")[0].split("]")[-1].strip())
    n_eval = int(out.split("targets,")[1].split("claims")[0].strip())
    assert n_eval >= n_targets
