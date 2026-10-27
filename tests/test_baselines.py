from pathlib import Path

import pytest

from agent.baselines import no_retrieval, parse_verdict, plain_rag
from core.cache import CacheStats, DiskCache
from core.llm import LLMClient, SpendLedger
from eval.datasets import Claim, label_counts, load_averitec, normalise_label, sample
from eval.metrics import score
from eval.run_baselines import main as run_main
from retrieval.bm25 import BM25Index
from retrieval.store import GoldEvidenceSource, KnowledgeStore

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CLAIMS_FILE = FIXTURES / "averitec_sample.json"
STORE_FILE = FIXTURES / "knowledge_store.jsonl"
OLLAMA = "http://localhost:11434/v1"


def make_llm(tmp_path, reply="Verdict: Supported\nReasoning: the evidence says so."):
    sent = []

    def transport(payload):
        sent.append(payload)
        return {
            "model": "qwen2.5:7b",
            "choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        }

    llm = LLMClient(base_url=OLLAMA, model="qwen2.5:7b", transport=transport)
    llm.cache = DiskCache("llm", root=tmp_path, stats=CacheStats())
    llm.ledger = SpendLedger(path=tmp_path, cap_usd=None)
    llm.sent = sent
    return llm


# --- label normalisation ------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Supported", "Supported"),
        ("Refuted", "Refuted"),
        ("Not Enough Evidence", "Not enough evidence"),
        ("Conflicting Evidence/Cherrypicking", "Conflicting evidence"),
        ("  supported  ", "Supported"),
        ("NEI", "Not enough evidence"),
    ],
)
def test_averitec_labels_map_to_our_verdicts(raw, expected):
    assert normalise_label(raw) == expected


def test_unmapped_label_is_an_error_not_a_silent_pass():
    with pytest.raises(ValueError, match="Unknown AVeriTeC label"):
        normalise_label("Mostly True")


def test_none_label_stays_none():
    assert normalise_label(None) is None


# --- dataset loading ----------------------------------------------------------------


def test_loads_the_fixture():
    claims = load_averitec(CLAIMS_FILE)
    assert len(claims) == 12
    assert all(c.claim and c.gold_label for c in claims)


def test_all_four_verdicts_present():
    assert set(label_counts(load_averitec(CLAIMS_FILE))) == {
        "Supported",
        "Refuted",
        "Not enough evidence",
        "Conflicting evidence",
    }


def test_sample_is_seeded_and_reproducible():
    claims = load_averitec(CLAIMS_FILE)
    assert [c.id for c in sample(claims, 5, seed=1)] == [c.id for c in sample(claims, 5, seed=1)]


def test_sample_larger_than_dataset_returns_everything():
    claims = load_averitec(CLAIMS_FILE)
    assert len(sample(claims, 999)) == len(claims)


def test_gold_evidence_pairs_question_with_answer():
    claim = load_averitec(CLAIMS_FILE)[0]
    evidence = claim.gold_evidence()
    assert evidence and evidence[0][0].startswith("https://")
    assert "Official records" in evidence[0][1]


# --- BM25 ---------------------------------------------------------------------------


def test_bm25_ranks_the_relevant_document_first():
    index = BM25Index(
        [("a", "penguins are flightless birds"), ("b", "the metro opened in 2017 in Kochi")]
    )
    assert index.search("when did the Kochi metro open", k=1)[0][0] == "b"


def test_bm25_excludes_on_request():
    index = BM25Index([("a", "metro rail"), ("b", "metro rail")])
    assert [i for i, _ in index.search("metro", k=5, exclude=lambda d: d == "a")] == ["b"]


def test_bm25_empty_index_is_safe():
    assert BM25Index([]).search("anything", k=3) == []


def test_bm25_returns_nothing_when_no_term_matches():
    assert BM25Index([("a", "penguins and krill")]).search("metro railway", k=3) == []


# --- knowledge store ----------------------------------------------------------------


def test_store_retrieves_relevant_passages():
    store = KnowledgeStore(STORE_FILE)
    hits = store.retrieve("0", load_averitec(CLAIMS_FILE)[0].claim, k=3)
    assert hits
    assert "metro rail" in hits[0].text.lower()


def test_store_blocks_fact_checking_domains(caplog):
    """Rule 5: the fixture plants a politifact page; it must never be returned."""
    store = KnowledgeStore(STORE_FILE)
    for claim in load_averitec(CLAIMS_FILE):
        hits = store.retrieve(str(claim.id), claim.claim, k=20)
        assert all("politifact" not in e.url for e in hits)


def test_store_reports_missing_claims():
    store = KnowledgeStore(STORE_FILE)
    assert store.has("0")
    assert not store.has("999")


def test_store_returns_empty_for_unknown_claim():
    assert KnowledgeStore(STORE_FILE).retrieve("999", "anything", k=5) == []


def test_store_tags_provenance():
    hits = KnowledgeStore(STORE_FILE).retrieve("1", "vaccine policy", k=2)
    assert all(e.retrieved_by == "knowledge_store" for e in hits)
    assert all(e.domain for e in hits)


def test_gold_source_is_tagged_as_annotation():
    claims = load_averitec(CLAIMS_FILE)
    hits = GoldEvidenceSource(claims).retrieve("0", "anything", k=5)
    assert hits and all(e.retrieved_by == "gold_annotation" for e in hits)


# --- verdict parsing ----------------------------------------------------------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("Verdict: Supported", "Supported"),
        ("Verdict: Refuted\nReasoning: no.", "Refuted"),
        ("verdict - not enough evidence", "Not enough evidence"),
        ("Verdict: Conflicting Evidence/Cherrypicking", "Conflicting evidence"),
        ("Verdict: NEI", "Not enough evidence"),
        ("I think this is clearly refuted by the sources.", "Refuted"),
    ],
)
def test_parses_verdicts(reply, expected):
    assert parse_verdict(reply).verdict == expected


def test_longest_alias_wins():
    """'Not enough evidence' must not be shadowed by a shorter alias."""
    assert parse_verdict("Verdict: Not enough evidence").verdict == "Not enough evidence"


def test_unparseable_falls_back_and_is_flagged():
    parsed = parse_verdict("I am unable to help with that.")
    assert parsed.verdict == "Not enough evidence"
    assert parsed.parsed is False


def test_rationale_is_extracted():
    parsed = parse_verdict("Verdict: Supported\nReasoning: Two sources agree on the date.")
    assert parsed.rationale == "Two sources agree on the date."


# --- the baselines ------------------------------------------------------------------


def test_no_retrieval_sends_no_evidence(tmp_path):
    llm = make_llm(tmp_path)
    traj = no_retrieval(load_averitec(CLAIMS_FILE)[0], llm)
    assert traj.evidence == []
    assert "Evidence:" not in llm.sent[0]["messages"][1]["content"]
    assert traj.source == "baseline_no_retrieval"


def test_plain_rag_puts_evidence_in_the_prompt(tmp_path):
    llm = make_llm(tmp_path)
    claim = load_averitec(CLAIMS_FILE)[0]
    traj = plain_rag(claim, llm, KnowledgeStore(STORE_FILE), k=3)
    # At most k, and never zero-scoring filler: the fixture's off-topic page must not
    # be padded in just to reach k.
    assert 1 <= len(traj.evidence) <= 3
    assert all("unrelated" not in e.url for e in traj.evidence)
    assert "metro rail" in traj.evidence[0].text.lower()
    assert "Evidence:" in llm.sent[0]["messages"][1]["content"]
    assert traj.source == "baseline_plain_rag"


def test_plain_rag_survives_a_claim_with_no_documents(tmp_path):
    llm = make_llm(tmp_path)
    orphan = Claim(id="999", claim="A claim with nothing in the store", gold_label="Refuted")
    traj = plain_rag(orphan, llm, KnowledgeStore(STORE_FILE), k=5)
    assert traj.evidence == []
    assert "(no evidence retrieved)" in llm.sent[0]["messages"][1]["content"]


def test_baseline_output_is_a_usable_trajectory(tmp_path):
    traj = no_retrieval(load_averitec(CLAIMS_FILE)[0], make_llm(tmp_path))
    assert traj.gold_label and traj.claim_id
    assert traj.correct is (traj.verdict == traj.gold_label)


def test_repeat_run_is_served_from_cache(tmp_path):
    llm = make_llm(tmp_path)
    claim = load_averitec(CLAIMS_FILE)[0]
    no_retrieval(claim, llm)
    no_retrieval(claim, llm)
    assert len(llm.sent) == 1, "rule 1: an uncached repeat call is a bug"


# --- metrics ------------------------------------------------------------------------


def test_perfect_predictions():
    m = score(["Supported", "Refuted"], ["Supported", "Refuted"])
    assert m.accuracy == 1.0 and m.macro_f1 == 1.0


def test_all_wrong():
    m = score(["Supported", "Refuted"], ["Refuted", "Supported"])
    assert m.accuracy == 0.0 and m.macro_f1 == 0.0


def test_macro_f1_ignores_absent_classes():
    m = score(["Supported"] * 3, ["Supported"] * 3)
    assert set(m.per_class) == {"Supported"}


def test_macro_f1_counts_a_predicted_but_never_correct_class():
    m = score(["Supported", "Supported"], ["Supported", "Refuted"])
    assert "Refuted" in m.per_class and m.per_class["Refuted"].f1 == 0.0


def test_confusion_matrix():
    m = score(["Supported", "Supported"], ["Supported", "Refuted"])
    assert m.confusion["Supported"]["Supported"] == 1
    assert m.confusion["Supported"]["Refuted"] == 1


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError, match="gold has"):
        score(["Supported"], [])


def test_empty_input_is_safe():
    assert score([], []).n == 0


def test_unparseable_count_is_carried_through():
    assert score(["Supported"], ["Supported"], unparseable=3).as_dict()["unparseable"] == 3


# --- the runner ---------------------------------------------------------------------


def test_dry_run_makes_no_calls(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VERIMEM_API_BASE_URL", OLLAMA)
    monkeypatch.setenv("VERIMEM_API_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("VERIMEM_RESULTS_DIR", str(tmp_path))
    assert run_main(["--data", str(CLAIMS_FILE), "--limit", "5", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "stopping before any call" in out
    assert "local endpoint, free" in out


def test_plain_rag_requires_a_store(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIMEM_API_BASE_URL", OLLAMA)
    monkeypatch.setenv("VERIMEM_API_MODEL", "qwen2.5:7b")
    with pytest.raises(SystemExit, match="--store is required"):
        run_main(["--data", str(CLAIMS_FILE), "--system", "plain_rag", "--limit", "2"])


def test_majority_baseline_is_the_floor():
    from eval.metrics import majority_baseline

    gold = ["Refuted"] * 6 + ["Supported"] * 4
    m = majority_baseline(gold)
    assert m.accuracy == 0.6
    # Always guessing one class cannot score well on macro-F1, which is the point.
    assert m.macro_f1 < m.accuracy


def test_majority_baseline_on_empty_input():
    from eval.metrics import majority_baseline

    assert majority_baseline([]).n == 0


# --- crag as a third system ---------------------------------------------------------


def test_crag_requires_a_trained_evaluator(tmp_path, monkeypatch):
    """Without one it would score with a lexical stub, and the CRAG row would
    measure word counting rather than the evaluator."""
    monkeypatch.setenv("VERIMEM_API_BASE_URL", OLLAMA)
    monkeypatch.setenv("VERIMEM_API_MODEL", "qwen2.5:7b")
    with pytest.raises(SystemExit, match="--evaluator is required"):
        run_main(
            [
                "--data",
                str(CLAIMS_FILE),
                "--store",
                str(STORE_FILE),
                "--system",
                "crag",
                "--limit",
                "2",
            ]
        )


def test_all_includes_three_systems():
    from eval.run_baselines import SYSTEMS

    assert set(SYSTEMS) == {"no_retrieval", "plain_rag", "crag"}


def test_both_still_means_the_two_baselines(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VERIMEM_API_BASE_URL", OLLAMA)
    monkeypatch.setenv("VERIMEM_API_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("VERIMEM_RESULTS_DIR", str(tmp_path))
    run_main(["--data", str(CLAIMS_FILE), "--store", str(STORE_FILE), "--limit", "3", "--dry-run"])
    assert "crag thresholds" not in capsys.readouterr().out
