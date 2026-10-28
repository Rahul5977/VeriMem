"""The adaptive adversary and the defence ablations (C1).

The first full sweep had `gated_memory` dropping 100% of poisoned records at every rate,
which looks like a decisive defence and is partly an artefact: the naive attack cites
domains with no history, and C3's prior puts a stranger below the trust floor. These
tests pin down the distinction so a later change cannot quietly erase it.
"""

from __future__ import annotations

import random

from attacks.memory import DEFAULT_DOMAINS, Target, fabricate, trusted_domains
from eval.run_poisoning import CONFIGS, build_trust
from experience.format import Evidence, Step, Trajectory
from experience.store import BetaTrust, ExperienceStore

FLOOR = 0.3


def _traj(claim: str, domain: str, correct: bool, i: int) -> Trajectory:
    return Trajectory(
        claim=claim,
        verdict="Refuted",
        rationale="because the record says so",
        evidence=[Evidence(url=f"https://{domain}/a/{i}", text="a passage of evidence")],
        steps=[Step("retrieve", "top-5")],
        claim_id=f"c{i}",
        # `correct` is derived from verdict == gold_label, so the label sets it.
        gold_label="Refuted" if correct else "Supported",
    )


def _history() -> list[Trajectory]:
    """A clean history where two publishers earned trust and one did not."""
    out = [_traj(f"claim about topic {i}", "reuters.com", True, i) for i in range(8)]
    out += [_traj(f"claim about topic {i}", "bbc.co.uk", True, i) for i in range(8, 12)]
    out += [_traj(f"claim about topic {i}", "randomblog.example", False, i) for i in range(12, 16)]
    return out


def test_trusted_domains_picks_the_established_ones() -> None:
    history = _history()
    picked = trusted_domains(history, build_trust(history), n=2)
    assert "reuters.com" in picked
    assert "randomblog.example" not in picked, "a domain with a failing record is not trusted"


def test_trusted_domains_skips_domains_without_a_track_record() -> None:
    history = _history()
    # One lucky hit should not make a domain the adversary's best impersonation target.
    history.append(_traj("a one-off claim", "fluke.example", True, 99))
    trust = build_trust(history)
    assert "fluke.example" not in trusted_domains(history, trust, n=3, min_observations=3)


def test_trusted_domains_falls_back_rather_than_attacking_with_nothing() -> None:
    """An empty domain pool would make harmless fakes and read as a defence that worked."""
    assert trusted_domains([], BetaTrust(), n=3) == DEFAULT_DOMAINS


def test_adaptive_fakes_clear_the_trust_floor_and_naive_ones_do_not() -> None:
    """The point of the exercise: C3 stops the naive attack, not the adaptive one."""
    history = _history()
    trust = build_trust(history)
    target = Target("t1", "The sky is green on Tuesdays.", "Refuted", "Supported")
    rng = random.Random(0)

    naive = fabricate(target, rng, DEFAULT_DOMAINS, 0)
    adaptive = fabricate(target, rng, trusted_domains(history, trust, n=2), 0)

    assert min(trust.trust(d) for d in naive.domains) < FLOOR
    assert min(trust.trust(d) for d in adaptive.domains) >= FLOOR


def test_trust_only_keeps_the_adaptive_attack_it_blocks_naively() -> None:
    """C3 alone: stops the stranger, waves the impersonator through."""
    history = _history()
    trust = build_trust(history)
    target = Target("t1", "The sky is green on Tuesdays.", "Refuted", "Supported")
    rng = random.Random(0)

    for domains, expected in ((DEFAULT_DOMAINS, 0), (trusted_domains(history, trust, n=2), 1)):
        fake = fabricate(target, rng, domains, 0)
        store = ExperienceStore([*history, fake], trust=trust, trust_floor=FLOOR)
        kept = store.retrieve(target.claim, claim_id=target.claim_id, k=5, use_trust=True)
        assert sum(1 for r in kept if r.trajectory.poisoned) == expected


def test_the_ablation_configs_isolate_one_defence_each() -> None:
    assert CONFIGS["memory"] == (True, False, False)
    assert CONFIGS["gate_only"] == (True, True, False)
    assert CONFIGS["trust_only"] == (True, False, True)
    assert CONFIGS["gated_memory"] == (True, True, True)
    # Every memory config must actually have retrieval on, or it is not a memory config.
    for name, (retrieval, _, _) in CONFIGS.items():
        assert retrieval is (name != "no_memory")
