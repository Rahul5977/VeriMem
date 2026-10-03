import pytest

from experience.format import (
    VERDICTS,
    Evidence,
    Step,
    Trajectory,
    estimate_tokens,
    read_jsonl,
    write_jsonl,
)


def make_traj(**kw) -> Trajectory:
    base = dict(
        claim="The Kochi metro opened in 2017",
        verdict="Supported",
        rationale="Two independent sources give the same opening date.",
        evidence=[
            Evidence(url="https://en.wikipedia.org/wiki/Kochi_Metro", text="Opened June 2017."),
            Evidence(url="https://www.thehindu.com/news/x", text="Inaugurated in June 2017."),
        ],
        steps=[Step("retrieve", "top-5"), Step("verify", "verdict produced")],
    )
    return Trajectory(**{**base, **kw})


def test_domain_is_derived_from_url():
    assert Evidence(url="https://www.bbc.co.uk/news/1").domain == "bbc.co.uk"


def test_explicit_domain_is_not_overwritten():
    assert Evidence(url="https://x.example/1", domain="custom").domain == "custom"


def test_rejects_an_unknown_verdict():
    with pytest.raises(ValueError, match="not one of"):
        make_traj(verdict="Mostly true")


def test_rejects_an_unknown_action():
    with pytest.raises(ValueError, match="not one of"):
        make_traj(action="refine-everything")


def test_correct_compares_against_gold():
    assert make_traj(gold_label="Supported").correct is True
    assert make_traj(gold_label="Refuted").correct is False


def test_correct_is_none_without_gold():
    assert make_traj().correct is None


def test_domains_are_deduped_in_order():
    traj = make_traj(
        evidence=[
            Evidence(url="https://a.org/1"),
            Evidence(url="https://b.org/1"),
            Evidence(url="https://a.org/2"),
        ]
    )
    assert traj.domains == ["a.org", "b.org"]


def test_render_answer_leads_with_the_verdict():
    assert make_traj().render_answer().startswith("Verdict: Supported")


def test_render_answer_without_rationale():
    assert make_traj(rationale="").render_answer() == "Verdict: Supported"


def test_context_render_respects_the_token_cap():
    traj = make_traj(
        evidence=[Evidence(url="https://a.org/1", text="word " * 4000) for _ in range(3)]
    )
    rendered = traj.render_for_context(token_cap=1000)
    assert estimate_tokens(rendered) <= 1000


def test_context_render_keeps_the_claim_even_at_a_tiny_cap():
    rendered = make_traj().render_for_context(token_cap=12)
    assert "Kochi" in rendered


def test_empty_evidence_renders_a_placeholder():
    assert make_traj(evidence=[]).render_evidence() == "(no evidence retrieved)"


def test_roundtrip_through_dict():
    traj = make_traj(gold_label="Supported", claim_id="c-1", action="correct")
    clone = Trajectory.from_dict(traj.to_dict())
    assert clone == traj


def test_from_dict_tolerates_the_derived_correct_field():
    data = make_traj(gold_label="Supported").to_dict()
    data["correct"] = True  # a writer that serialised the property
    assert Trajectory.from_dict(data).correct is True


def test_unknown_keys_are_kept_in_metadata():
    data = make_traj().to_dict()
    data["experimental_field"] = 7
    assert Trajectory.from_dict(data).metadata["experimental_field"] == 7


def test_jsonl_roundtrip(tmp_path):
    trajs = [make_traj(claim=f"claim {i}") for i in range(5)]
    path = write_jsonl(trajs, tmp_path / "t.jsonl")
    assert read_jsonl(path) == trajs


def test_bad_line_names_the_file_and_line(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"claim": "a", "verdict": "Supported"}\n{"claim": "b"}\n')
    with pytest.raises(ValueError, match=r"t\.jsonl:2"):
        read_jsonl(path)


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "t.jsonl"
    write_jsonl([make_traj()], path)
    path.write_text(path.read_text() + "\n\n")
    assert len(read_jsonl(path)) == 1


def test_every_verdict_is_constructible():
    for verdict in VERDICTS:
        assert make_traj(verdict=verdict).verdict == verdict
