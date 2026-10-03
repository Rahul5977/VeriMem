from core.config import BASELINE, FULL_VERIMEM, RunConfig


def test_default_is_the_baseline():
    cfg = RunConfig()
    assert not cfg.experience_retrieval
    assert not cfg.gate_experience
    assert not cfg.source_trust
    assert cfg.verifier == "prompted"


def test_full_system_turns_every_flag_on():
    assert all(
        [FULL_VERIMEM.experience_retrieval, FULL_VERIMEM.gate_experience, FULL_VERIMEM.source_trust]
    )
    assert FULL_VERIMEM.verifier == "finetuned"


def test_roundtrip_through_yaml(tmp_path):
    cfg = RunConfig(experience_retrieval=True, seed=7, limit=20, notes="pilot")
    cfg.save(tmp_path)
    assert RunConfig.load(tmp_path / "config.yaml") == cfg


def test_unknown_keys_land_in_extra():
    cfg = RunConfig.from_dict({"seed": 3, "mystery": 42})
    assert cfg.seed == 3 and cfg.extra["mystery"] == 42


def test_with_returns_a_copy():
    assert BASELINE.with_(seed=99).seed == 99
    assert BASELINE.seed == 13


def test_student_is_never_the_teacher():
    """Rule 8: Qwen2.5-3B is the student; the teacher is a hosted model."""
    cfg = RunConfig()
    assert "Qwen2.5-3B" not in cfg.api_model


def test_sft_context_fits_the_8gb_budget():
    """k trajectories at the token cap must leave room under max_seq_length."""
    cfg = RunConfig()
    assert cfg.sft_k_trajectories <= 2
    assert cfg.sft_k_trajectories * cfg.trajectory_token_cap < cfg.max_seq_length
