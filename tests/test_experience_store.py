import random

from attacks.memory import DEFAULT_DOMAINS, Target, fabricate
from evaluator.score import LexicalScorer
from experience.format import Evidence, Trajectory
from experience.store import BetaTrust, ExperienceStore, UniformTrust, is_self


def traj(claim, verdict="Supported", cid=None, domains=("bbc.co.uk",), correct=None):
    gold = verdict if correct is True else ("Refuted" if correct is False else None)
    if verdict == "Refuted" and correct is False:
        gold = "Supported"
    return Trajectory(
        claim=claim,
        verdict=verdict,
        claim_id=cid,
        gold_label=gold,
        evidence=[Evidence(url=f"https://{d}/a", text=f"evidence about {claim}") for d in domains],
    )


class FixedScorer:
    """Returns a constant, so a gate test measures the gate and not BM25.
    With the lexical stub the similarity and the gate score move together, and a
    trajectory BM25 never returns cannot demonstrate anything about gating."""

    def __init__(self, value):
        self.value = value

    def score(self, claim, passages):
        return [self.value] * len(passages)


def store_of(*trajectories, **kw):
    return ExperienceStore(list(trajectories), **kw)


# --- plain retrieval ----------------------------------------------------------------


def test_retrieves_the_similar_trajectory():
    s = store_of(traj("The Kochi metro opened in 2017"), traj("Penguins are flightless birds"))
    got = s.retrieve("When did the Kochi metro open?", k=1)
    assert len(got) == 1 and "Kochi metro" in got[0].trajectory.claim


def test_respects_k():
    s = store_of(*[traj(f"metro claim number {i}") for i in range(10)])
    assert len(s.retrieve("metro claim", k=3)) == 3


def test_empty_store_returns_nothing():
    assert store_of().retrieve("anything", k=3) == []


def test_k_zero_returns_nothing():
    assert store_of(traj("a metro claim")).retrieve("metro", k=0) == []


# --- self-exclusion (rule 5) --------------------------------------------------------


def test_excludes_by_claim_id():
    s = store_of(traj("The metro opened in 2017", cid="c1"))
    assert s.retrieve("The metro opened in 2017", claim_id="c1", k=3) == []


def test_excludes_by_claim_text_even_under_a_different_id():
    """A poisoned record carries the target's wording under a new id."""
    s = store_of(traj("The metro opened in 2017", cid="poison-1"))
    assert s.retrieve("The metro opened in 2017", claim_id="c1", k=3) == []


def test_is_self_matches_text_case_insensitively():
    assert is_self("Same Claim", "c1", traj("  same claim ", cid="c2"))


def test_a_different_claim_is_not_self():
    s = store_of(traj("The metro extension opened in 2019", cid="c2"))
    assert len(s.retrieve("The metro opened in 2017", claim_id="c1", k=3)) == 1


# --- C2: evaluator gating -----------------------------------------------------------


def poisoned(target_claim, n=1):
    rng = random.Random(1)
    t = Target("t1", target_claim, "Refuted", "Supported")
    return [fabricate(t, rng, DEFAULT_DOMAINS, i) for i in range(n)]


def test_gate_drops_low_scoring_trajectories():
    s = store_of(
        traj("The Kochi metro opened in 2017"), scorer=FixedScorer(-0.8), gate_threshold=0.0
    )
    assert s.retrieve("When did the Kochi metro open?", k=3, gate=True) == []
    assert s.stats.dropped_by_gate == 1


def test_gate_keeps_relevant_trajectories():
    s = store_of(
        traj("The Kochi metro opened to the public in June 2017"),
        scorer=LexicalScorer(),
        gate_threshold=-0.9,
    )
    assert len(s.retrieve("When did the Kochi metro open?", k=3, gate=True)) == 1


def test_gate_is_off_unless_asked():
    s = store_of(
        traj("The Kochi metro opened in 2017"), scorer=FixedScorer(-0.8), gate_threshold=0.0
    )
    assert len(s.retrieve("Kochi metro opening", k=3, gate=False)) == 1
    assert s.stats.dropped_by_gate == 0


def test_gate_records_the_score_it_used():
    s = store_of(
        traj("The Kochi metro opened in 2017"), scorer=LexicalScorer(), gate_threshold=-1.0
    )
    got = s.retrieve("Kochi metro opening", k=1, gate=True)
    assert got[0].gate_score is not None


def test_over_fetches_so_gating_does_not_shrink_the_context():
    """C2 should change what is retrieved, not how much, or the comparison confounds
    quality with quantity."""
    good = [traj(f"The Kochi metro opened in 201{i}") for i in range(5)]
    noise = [traj(f"Penguins story number {i}") for i in range(5)]
    s = store_of(*(noise + good), scorer=LexicalScorer(), gate_threshold=0.0)
    assert len(s.retrieve("When did the Kochi metro open?", k=3, gate=True)) == 3


# --- C3: trust ----------------------------------------------------------------------


def test_trust_drops_low_trust_sources():
    trust = BetaTrust()
    for _ in range(10):
        trust.update("badsite.com", was_correct=False)
    s = store_of(traj("a metro claim", domains=("badsite.com",)), trust=trust, trust_floor=0.3)
    assert s.retrieve("metro claim", k=3, use_trust=True) == []
    assert s.stats.dropped_by_trust == 1


def test_trust_keeps_reliable_sources():
    trust = BetaTrust()
    for _ in range(10):
        trust.update("bbc.co.uk", was_correct=True)
    s = store_of(traj("a metro claim", domains=("bbc.co.uk",)), trust=trust, trust_floor=0.3)
    assert len(s.retrieve("metro claim", k=3, use_trust=True)) == 1


def test_unseen_domain_sits_at_one_half():
    assert BetaTrust().trust("never-seen.com") == 0.5


def test_trust_moves_with_evidence():
    t = BetaTrust()
    t.update("a.com", True)
    t.update("a.com", True)
    t.update("b.com", False)
    t.update("b.com", False)
    assert t.trust("a.com") > 0.5 > t.trust("b.com")


def test_observe_credits_every_domain_of_a_correct_verification():
    t = BetaTrust()
    t.observe(traj("x", domains=("a.com", "b.com"), correct=True))
    assert t.trust("a.com") > 0.5 and t.trust("b.com") > 0.5


def test_observe_ignores_unlabelled_trajectories():
    t = BetaTrust()
    t.observe(traj("x", domains=("a.com",)))
    assert t.as_dict() == {}


def test_uniform_trust_is_explicit_not_implicit():
    """A run that silently used uniform trust while claiming to test C3 would report a
    false negative on H2."""
    assert UniformTrust().trust("anything") == 1.0


# --- what the poisoning study needs to report ---------------------------------------


def test_stats_count_poisoned_records_separately():
    fakes = poisoned("The Kochi metro opened in 2017", n=3)
    s = store_of(traj("Something about penguins entirely", cid="other"), *fakes)
    s.retrieve("The Kochi metro opened in 2017", claim_id="target", k=5)
    assert s.stats.poisoned_considered == 3
    assert s.stats.poisoned_kept == 3
    assert s.stats.as_dict()["poison_pass_rate"] == 1.0


def test_a_verbatim_fake_is_excluded_but_a_paraphrased_one_is_not():
    """Exact copying is defeated for free by self-exclusion, so the attack paraphrases.
    This pins that the paraphrase actually survives."""
    target_text = "The Kochi metro opened in 2017"
    fakes = poisoned(target_text, n=2)
    s = store_of(*fakes)
    got = s.retrieve(target_text, claim_id="target", k=5)
    assert got, "paraphrased fakes must stay retrievable"
    assert all(r.trajectory.poisoned for r in got)


def test_poison_pass_rate_is_none_when_nothing_poisoned():
    s = store_of(traj("a claim"))
    s.retrieve("a claim about something", k=3)
    assert s.stats.as_dict()["poison_pass_rate"] is None


def test_poisoned_count_is_visible():
    s = store_of(traj("real"), *poisoned("fake target", n=2))
    assert s.poisoned_count == 2 and len(s) == 3


# --- rendering ----------------------------------------------------------------------


def test_render_produces_a_context_block():
    s = store_of(traj("The Kochi metro opened in 2017"))
    got = s.retrieve("Kochi metro", k=1)
    block = s.render(got)
    assert "Past verifications" in block and "Kochi metro" in block


def test_render_of_nothing_is_empty():
    assert store_of().render([]) == ""
