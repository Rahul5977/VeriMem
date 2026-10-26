import json

import pytest

from agent.crag import grade_evidence, rewrite_query, route, run_crag
from core.cache import CacheStats, DiskCache
from core.config import RunConfig
from core.llm import LLMClient, SpendLedger
from eval.datasets import Claim
from evaluator.score import CachedScorer, LexicalScorer
from experience.format import Evidence
from retrieval.refine import recompose, refine, split_strips, strips_from_evidence
from retrieval.store import PooledKnowledgeStore

OLLAMA = "http://localhost:11434/v1"


def make_llm(tmp_path, reply="Verdict: Supported\nReasoning: yes."):
    sent = []

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


def pool(tmp_path, rows):
    p = tmp_path / "pool.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return PooledKnowledgeStore(p)


def claim(text="Did the Kochi metro open in 2017?", cid="c1"):
    return Claim(id=cid, claim=text, gold_label="Supported")


# --- strips -------------------------------------------------------------------------


def test_splits_on_sentences():
    assert len(split_strips("The metro opened in 2017. It cost a lot of money overall.")) == 2


def test_does_not_split_on_abbreviations():
    s = split_strips("Dr. Smith confirmed the opening date of the metro line in Kochi.")
    assert len(s) == 1


def test_drops_fragments_too_short_to_be_evidence():
    assert split_strips("Home. Next. The metro opened in June 2017 after long delays.") == [
        "The metro opened in June 2017 after long delays."
    ]


def test_strips_keep_their_source():
    ev = [Evidence(url="https://bbc.co.uk/a", text="The metro opened in 2017 in Kochi city.")]
    s = strips_from_evidence(ev)[0]
    assert s.domain == "bbc.co.uk" and "bbc.co.uk" in s.render()


# --- refinement ---------------------------------------------------------------------


def test_refine_ranks_relevant_strips_first():
    ev = [
        Evidence(
            url="https://a.org/1",
            text=(
                "Penguins are flightless birds of the southern hemisphere. "
                "The Kochi metro opened to the public in June 2017."
            ),
        )
    ]
    out = refine("When did the Kochi metro open?", ev, LexicalScorer(), top_k=2)
    assert "Kochi metro opened" in out[0].text


def test_refine_respects_top_k():
    ev = [
        Evidence(
            url="https://a.org/1",
            text=" ".join(f"Sentence number {i} about the Kochi metro opening." for i in range(10)),
        )
    ]
    assert len(refine("Kochi metro", ev, LexicalScorer(), top_k=3)) <= 3


def test_refine_can_return_nothing_when_everything_is_irrelevant():
    """The honest answer when retrieval found only noise -- and what sends a claim
    down the rewrite branch."""
    ev = [Evidence(url="https://a.org/1", text="Penguins are flightless birds of the south.")]
    assert refine("Kochi metro opening date", ev, LexicalScorer(), min_score=0.0) == []


def test_refine_on_no_evidence():
    assert refine("anything", [], LexicalScorer()) == []


def test_recompose_marks_the_empty_case():
    assert "no relevant evidence" in recompose([])


# --- grading and routing ------------------------------------------------------------


def test_grade_uses_the_best_passage_not_the_average():
    """One good document is enough; averaging lets noise bury it."""
    ev = [
        Evidence(url="https://a.org/1", text="Penguins are flightless birds entirely."),
        Evidence(url="https://b.org/1", text="The Kochi metro opened in June 2017 officially."),
        Evidence(url="https://c.org/1", text="Unrelated text about cooking recipes here."),
    ]
    good_only = grade_evidence("Kochi metro opened", [ev[1]], LexicalScorer())
    mixed = grade_evidence("Kochi metro opened", ev, LexicalScorer())
    assert mixed == pytest.approx(good_only)


def test_grade_of_nothing_is_the_floor():
    assert grade_evidence("x", [], LexicalScorer()) == -1.0


@pytest.mark.parametrize(
    "grade,expected",
    [
        (0.9, "correct"),
        (0.6, "correct"),
        (-0.9, "incorrect"),
        (-0.6, "incorrect"),
        (0.0, "ambiguous"),
        (0.59, "ambiguous"),
        (-0.59, "ambiguous"),
    ],
)
def test_three_way_routing(grade, expected):
    assert route(grade, upper=0.6, lower=-0.6) == expected


# --- query rewriting ----------------------------------------------------------------


def test_rewrite_returns_a_short_query(tmp_path):
    llm = make_llm(tmp_path, reply="kochi metro opening date 2017")
    assert rewrite_query(claim().claim, llm) == "kochi metro opening date 2017"


def test_rewrite_strips_quotes_and_whitespace(tmp_path):
    llm = make_llm(tmp_path, reply='  "kochi metro 2017"  \n')
    assert rewrite_query(claim().claim, llm) == "kochi metro 2017"


def test_rewrite_falls_back_to_the_claim_when_the_model_says_nothing(tmp_path):
    llm = make_llm(tmp_path, reply="   ")
    assert rewrite_query(claim().claim, llm) == claim().claim


def test_rewrite_truncates_a_rambling_reply(tmp_path):
    llm = make_llm(tmp_path, reply="word " * 200)
    assert len(rewrite_query(claim().claim, llm)) <= 200


# --- the pipeline -------------------------------------------------------------------


GOOD = {
    "url": "https://a.org/1",
    "url2text": ["The Kochi metro opened to the public in June 2017 after several delays."] * 4,
}
NOISE = {"url": "https://b.org/1", "url2text": ["Penguins are flightless birds."] * 4}


def test_good_evidence_routes_to_correct_and_never_rewrites(tmp_path):
    llm = make_llm(tmp_path)
    cfg = RunConfig(crag_upper_threshold=0.2, crag_lower_threshold=-0.6)
    r = run_crag(claim(), llm, pool(tmp_path, [GOOD]), LexicalScorer(), cfg)
    assert r.action == "correct"
    assert r.rewritten_query is None
    assert not any(s.action == "rewrite" for s in r.steps)


def test_bad_evidence_routes_to_incorrect_and_rewrites(tmp_path):
    llm = make_llm(tmp_path, reply="kochi metro opening")
    cfg = RunConfig(crag_upper_threshold=0.9, crag_lower_threshold=0.5)
    r = run_crag(claim(), llm, pool(tmp_path, [NOISE]), LexicalScorer(), cfg)
    assert r.action == "incorrect"
    assert r.rewritten_query is not None
    assert any(s.action == "rewrite" for s in r.steps)


def test_ambiguous_does_both(tmp_path):
    llm = make_llm(tmp_path, reply="kochi metro")
    cfg = RunConfig(crag_upper_threshold=0.99, crag_lower_threshold=-0.99)
    r = run_crag(claim(), llm, pool(tmp_path, [GOOD, NOISE]), LexicalScorer(), cfg)
    assert r.action == "ambiguous"
    actions = [s.action for s in r.steps]
    assert "refine" in actions and "rewrite" in actions


def test_every_run_records_a_full_trace(tmp_path):
    llm = make_llm(tmp_path)
    r = run_crag(claim(), llm, pool(tmp_path, [GOOD]), LexicalScorer())
    assert [s.action for s in r.steps][:3] == ["retrieve", "grade", "route"]
    assert r.steps[-1].action == "verify"


def test_empty_store_does_not_crash_and_grades_at_the_floor(tmp_path):
    llm = make_llm(tmp_path, reply="kochi metro")
    r = run_crag(claim(), llm, pool(tmp_path, []), LexicalScorer())
    assert r.grade == -1.0 and r.action == "incorrect"


def test_crag_emits_a_usable_trajectory(tmp_path):
    from agent.crag import crag

    t = crag(claim(), make_llm(tmp_path), pool(tmp_path, [GOOD]), LexicalScorer())
    assert t.source == "crag" and t.action in ("correct", "incorrect", "ambiguous")
    assert t.gold_label == "Supported"
    assert t.metadata["grade"] is not None


# --- the cached scorer --------------------------------------------------------------


def test_scorer_cache_avoids_rescoring(tmp_path):
    calls = []

    class Counting(LexicalScorer):
        def score(self, claim, passages):
            calls.append(len(passages))
            return super().score(claim, passages)

    s = CachedScorer(Counting())
    s.cache = DiskCache("evaluator", root=tmp_path, stats=CacheStats())
    s.score("a claim", ["passage one", "passage two"])
    s.score("a claim", ["passage one", "passage two"])
    assert calls == [2], "rule 1: rescoring identical pairs is a bug"


def test_scorer_cache_only_computes_the_missing_ones(tmp_path):
    calls = []

    class Counting(LexicalScorer):
        def score(self, claim, passages):
            calls.append(len(passages))
            return super().score(claim, passages)

    s = CachedScorer(Counting())
    s.cache = DiskCache("evaluator", root=tmp_path, stats=CacheStats())
    s.score("a claim", ["one"])
    s.score("a claim", ["one", "two"])
    assert calls == [1, 1]
