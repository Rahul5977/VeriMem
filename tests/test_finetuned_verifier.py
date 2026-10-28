"""Serving a fine-tuned verifier (P2.11, C4).

No GPU here: the generation step is stubbed so the parts that are easy to get wrong --
which prompt is sent, what the cache is keyed on, where the answer is cut from the output
-- are tested on the Mac. The model loading itself is only exercised on Kaggle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent.baselines import SYSTEM, parse_verdict
from core.cache import DiskCache
from core.config import RunConfig
from core.prompts import VERIFIER_SYSTEM
from training.verifier import FinetunedVerifier, load_verifier


class _Tokenizer:
    """Just enough of a tokenizer to see what the model was handed."""

    pad_token = "<pad>"
    pad_token_id = 0

    def __init__(self) -> None:
        self.seen: list[Any] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.seen.append(messages)
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)

    def __call__(self, text, return_tensors=None):
        return _Encoding(text)

    def decode(self, ids, skip_special_tokens=True):
        return str(ids)


class _Encoding(dict):
    def __init__(self, text: str) -> None:
        super().__init__(input_ids=_Ids(len(text.split("|"))))

    def to(self, _device):
        return self


class _Ids:
    def __init__(self, n: int) -> None:
        self.shape = (1, n)

    def __getitem__(self, _i):
        return self


class _StubVerifier(FinetunedVerifier):
    """Counts real generations so cache hits can be told from fresh work."""

    def __init__(
        self, *args: Any, answer: str = "Verdict: Supported\nReasoning: it checks out.", **kw: Any
    ) -> None:
        super().__init__(*args, **kw)
        self.answer = answer
        self.generations = 0

    def load(self) -> None:
        self._tokenizer = _Tokenizer()
        self._model = object()

    def chat(self, messages, *, max_tokens=None, **kw):
        # Exercise the real caching path, stub only the generation inside it.
        # The real key builder, so a change to it is caught here.
        key = self.cache_key(messages, max_tokens or self.max_new_tokens)
        hit = self.cache.get(key)
        if hit is not None:
            from core.llm import LLMResponse

            return LLMResponse(text=hit["text"], model=self.model, cached=True)
        self.load()
        self._tokenizer.apply_chat_template(list(messages))
        self.generations += 1
        self.cache.set(key, {"text": self.answer})
        from core.llm import LLMResponse

        return LLMResponse(text=self.answer, model=self.model)


@pytest.fixture
def adapter(tmp_path: Path) -> Path:
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct"})
    )
    (d / "adapter_model.safetensors").write_bytes(b"x" * 64)
    return d


def _mk(adapter: Path, tmp_path: Path, **kw: Any) -> _StubVerifier:
    return _StubVerifier(adapter, cache=DiskCache("verifier-test", root=tmp_path / "cache"), **kw)


def test_base_model_comes_from_the_adapters_own_config(adapter: Path, tmp_path: Path) -> None:
    """A mismatched base silently produces garbage, so it is never guessed."""
    assert _mk(adapter, tmp_path).base_model == "Qwen/Qwen2.5-3B-Instruct"


def test_a_missing_adapter_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No adapter at"):
        FinetunedVerifier(tmp_path / "nope")


def test_no_adapter_means_use_the_teacher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERIMEM_VERIFIER_ADAPTER", raising=False)
    assert load_verifier(None) is None


def test_asking_for_finetuned_without_an_adapter_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falling back silently would report a fine-tuned row that was never fine-tuned."""
    monkeypatch.delenv("VERIMEM_VERIFIER_ADAPTER", raising=False)
    with pytest.raises(SystemExit, match="no adapter was given"):
        load_verifier(None, RunConfig(verifier="finetuned"))


def test_complete_sends_the_prompt_the_adapter_was_trained_on(
    adapter: Path, tmp_path: Path
) -> None:
    """The whole point of core.prompts: trained string == served string."""
    v = _mk(adapter, tmp_path)
    v.complete("Claim: x\n\nEvidence:\ny", system=SYSTEM)
    messages = v._tokenizer.seen[-1]
    assert messages[0] == {"role": "system", "content": VERIFIER_SYSTEM}
    assert messages[1]["content"].startswith("Claim: x")


def test_a_repeated_call_is_served_from_cache(adapter: Path, tmp_path: Path) -> None:
    """Rule 1. An uncached repeat is a bug, and a bulk eval repeats constantly."""
    v = _mk(adapter, tmp_path)
    first = v.complete("Claim: x", system=SYSTEM)
    second = v.complete("Claim: x", system=SYSTEM)
    assert v.generations == 1
    assert not first.cached
    assert second.cached
    assert second.text == first.text


def test_retraining_into_the_same_path_invalidates_the_cache(adapter: Path, tmp_path: Path) -> None:
    """Otherwise a second adapter is evaluated with the first one's answers."""
    cache = DiskCache("verifier-test", root=tmp_path / "cache")
    before = _StubVerifier(adapter, cache=cache, answer="Verdict: Supported")
    before.complete("Claim: x", system=SYSTEM)

    (adapter / "adapter_model.safetensors").write_bytes(b"y" * 128)  # new weights, same path
    after = _StubVerifier(adapter, cache=cache, answer="Verdict: Refuted")
    reply = after.complete("Claim: x", system=SYSTEM)

    assert after.generations == 1, "the new adapter must not reuse the old one's answer"
    assert reply.text == "Verdict: Refuted"


def test_the_answer_parses_with_the_agents_own_parser(adapter: Path, tmp_path: Path) -> None:
    """The verifier is only useful if the pipeline can read what it says."""
    v = _mk(adapter, tmp_path)
    parsed = parse_verdict(v.complete("Claim: x", system=SYSTEM).text)
    assert parsed.verdict == "Supported"
    assert parsed.parsed


def test_the_model_name_identifies_the_run_that_produced_it(adapter: Path, tmp_path: Path) -> None:
    """It lands in every trajectory's metadata, so it has to say which adapter answered."""
    assert _mk(adapter, tmp_path).model.endswith(adapter.parent.name)
