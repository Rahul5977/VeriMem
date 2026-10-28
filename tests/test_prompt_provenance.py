"""The verifier prompt is part of the cache key, so it is part of a run's identity.

This exists because of a specific incident. The prompt was defined twice -- once in
`training.data` for building SFT examples, once in `agent.baselines` for inference -- and
the two drifted by a few words. Unifying them was the right fix and it also invalidated
every cached verify call and made every number already on disk unreproducible from the
current code, with nothing in any results folder recording which prompt had produced it.

These tests do not stop the prompt from changing. They make a change impossible to make
*silently*.
"""

from __future__ import annotations

from pathlib import Path

from agent.baselines import SYSTEM
from core.config import RunConfig
from core.prompts import VERIFIER_SYSTEM, prompt_fingerprint, verifier_user_message
from experience.format import Evidence, Step, Trajectory
from training.data import SYSTEM_PROMPT, build_example


def test_training_and_inference_use_one_string() -> None:
    """The original bug. A model fine-tuned on one prompt must not be served another."""
    assert SYSTEM is VERIFIER_SYSTEM
    assert SYSTEM_PROMPT is VERIFIER_SYSTEM


def test_the_training_prompt_renders_the_same_user_turn_as_inference() -> None:
    """Agreeing on the system message is not enough if the user turn differs."""
    traj = Trajectory(
        claim="A claim about a measurable fact.",
        verdict="Refuted",
        rationale="r",
        evidence=[Evidence(url="https://example.com/1", text="A retrieved passage.")],
        steps=[Step("retrieve", "top-5")],
        claim_id="c1",
        gold_label="Refuted",
    )
    user = build_example(traj).prompt[1]["content"]
    assert user.startswith("Claim: A claim about a measurable fact.")
    # The same renderer the agent uses, with the agent's own section order.
    assert user == verifier_user_message(traj.claim, user.split("Evidence:\n", 1)[1])


def test_the_fingerprint_changes_with_the_prompt() -> None:
    import core.prompts as prompts

    before = prompt_fingerprint()
    original = prompts.VERIFIER_SYSTEM
    try:
        prompts.VERIFIER_SYSTEM = original + " One more sentence."
        assert prompt_fingerprint() != before
    finally:
        prompts.VERIFIER_SYSTEM = original
    assert prompt_fingerprint() == before


def test_a_saved_config_records_which_prompt_produced_it(tmp_path: Path) -> None:
    import yaml

    path = RunConfig(split="dev").save(tmp_path)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["verifier_prompt_sha"] == prompt_fingerprint()


def test_a_run_from_todays_prompt_is_recognised(tmp_path: Path) -> None:
    RunConfig(split="dev").save(tmp_path)
    assert RunConfig.load(tmp_path / "config.yaml").prompt_matches_code is True


def test_a_run_from_a_different_prompt_is_flagged(tmp_path: Path) -> None:
    """The whole point: a stale results folder must announce itself rather than be
    quietly compared against a fresh run."""
    import yaml

    path = RunConfig(split="dev").save(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["verifier_prompt_sha"] = "000000000000"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    loaded = RunConfig.load(path)
    assert loaded.prompt_matches_code is False


def test_a_config_predating_the_stamp_is_unknown_not_matching(tmp_path: Path) -> None:
    """Every folder written before today has no stamp. "Unknown" must not read as "fine"."""
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"split": "dev"}), encoding="utf-8")
    assert RunConfig.load(path).prompt_matches_code is None


def test_provenance_stays_out_of_config_identity(tmp_path: Path) -> None:
    """It describes the code, not the experiment, so it must not change equality, the
    serialised config, or the cache key."""
    cfg = RunConfig(split="dev")
    cfg.save(tmp_path)
    reloaded = RunConfig.load(tmp_path / "config.yaml")

    assert reloaded == cfg
    assert "verifier_prompt_sha" not in cfg.to_dict()
    assert "verifier_prompt_sha" not in reloaded.to_dict()
    assert "verifier_prompt_sha" not in reloaded.extra


def test_the_system_message_is_part_of_the_cache_key() -> None:
    """Why a prompt change invalidates everything: this is the mechanism.

    `LLMClient.chat` keys on the message list, system turn included, so editing the
    prompt turns every stored verify call into a miss.
    """
    from core.cache import hash_payload

    same = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    with_a = {"model": "m", "messages": [{"role": "system", "content": "A"}, same["messages"][0]]}
    with_b = {"model": "m", "messages": [{"role": "system", "content": "B"}, same["messages"][0]]}

    assert hash_payload(with_a) != hash_payload(with_b)
    assert hash_payload(with_a) != hash_payload(same)
