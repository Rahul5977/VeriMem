"""Serve a LoRA-fine-tuned verifier (P2.11, C4).

`RunConfig.verifier` has had a `"finetuned"` option since Phase 0 and nothing
implemented it, so the two SFT runs produced adapters that could not be evaluated --
and C4's four-row table had no path to a number. This is that path.

**It impersonates `LLMClient`.** The agent's verify step is a single
`llm.complete(prompt, system=...)`, so a fine-tuned verifier needs no agent changes, only
the same method returning the same `LLMResponse`. That keeps the comparison honest: the
prompted and fine-tuned rows differ in the weights that answer, not in the pipeline.

**Query rewriting stays with the teacher.** The student was fine-tuned on one task --
claim plus evidence to verdict -- and asking it to rewrite a search query would be asking
for behaviour no training example contained. So `run_crag` takes a separate `verifier`:
the teacher keeps rewriting and decomposing, and only the final judgement moves.

**Prompts come from `core.prompts`**, which is the same string the adapter was trained
on. See that module for why it is not defined here.

Torch, transformers and peft are imported lazily inside `load`, so this module can be
imported on the Mac (rule: guard GPU imports) and only a real load needs a GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.cache import DiskCache, get_cache
from core.config import RunConfig
from core.llm import LLMResponse

# The answer is "Verdict: <label>" plus a sentence or two. Generating far past that is
# pure latency on a bulk run, and a fine-tuned model that will not stop is a finding
# worth seeing rather than hiding behind a large budget.
DEFAULT_MAX_NEW_TOKENS = 128


@dataclass
class FinetunedVerifier:
    """The student verifier -- a 4-bit base, optionally with a LoRA adapter over it.

    **`adapter=None` is the prompted row of the G2 table, not a degenerate case.** The
    prompted and fine-tuned rows have to travel the same serving path or the comparison
    measures quantisation, chat templating and decoding settings alongside the weights.
    Routing the base model through Ollama instead would have made every one of those
    differ silently.

    Deterministic by default (greedy), because the evaluation compares verdicts across
    configurations and sampling noise would be indistinguishable from a real effect.
    """

    adapter: Path | None
    base_model: str = ""
    load_in_4bit: bool = True
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    cache: DiskCache | None = None
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.adapter is not None:
            self.adapter = Path(self.adapter)
            if not self.adapter.exists():
                raise SystemExit(
                    f"No adapter at {self.adapter}. Train one with `python -m training.sft` "
                    "and point --verifier-adapter at its `adapter/` directory."
                )
        if not self.base_model:
            self.base_model = (
                _base_from_adapter(self.adapter) if self.adapter else None
            ) or RunConfig().verifier_model
        if self.cache is None:
            self.cache = get_cache("verifier")

    # -- loading ----------------------------------------------------------------

    def load(self) -> None:
        """Pull the base model and apply the adapter. Idempotent."""
        if self._model is not None:
            return

        import torch  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        kwargs: dict[str, Any] = {"device_map": "auto"}
        if self.load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            # fp16 compute, not bf16: a T4 has no bf16, and core.device already makes
            # this choice everywhere else in the project.
            compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute,
                bnb_4bit_use_double_quant=True,
            )

        base = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)
        self._model = PeftModel.from_pretrained(base, str(self.adapter)) if self.adapter else base
        self._model.eval()

        # The adapter directory carries the tokenizer the run actually trained with,
        # including any added tokens; the hub copy may not match it.
        src = self.base_model
        if self.adapter and (self.adapter / "tokenizer_config.json").exists():
            src = self.adapter
        self._tokenizer = AutoTokenizer.from_pretrained(str(src))
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    # -- the LLMClient surface --------------------------------------------------

    @property
    def model(self) -> str:
        """Lands in every trajectory's metadata, so it must name what actually answered."""
        if self.adapter is None:
            return f"{self.base_model} (prompted, no adapter)"
        return f"{self.base_model}+{self.adapter.parent.name}"

    def complete(self, prompt: str, system: str | None = None, **kwargs: Any) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def cache_key(self, messages: Any, budget: int) -> dict[str, Any]:
        """What makes two calls the same call.

        The adapter fingerprint is part of it: retraining into the same output directory
        would otherwise serve the previous adapter's answers and silently evaluate the
        wrong model.
        """
        return {
            "base": self.base_model,
            "adapter": str(self.adapter) if self.adapter else None,
            "adapter_mtime": _adapter_fingerprint(self.adapter),
            "messages": list(messages),
            "max_tokens": budget,
            "greedy": True,
        }

    def chat(self, messages: Any, *, max_tokens: int | None = None, **_: Any) -> LLMResponse:
        """One verdict. Cached on disk like every other model call (rule 1).

        Extra decoding kwargs are accepted and ignored: callers written against
        `LLMClient` pass `temperature` and `seed`, and generation here is greedy, so
        honouring them would be a lie and rejecting them would break the drop-in.
        """
        budget = max_tokens or self.max_new_tokens
        key = self.cache_key(messages, budget)
        assert self.cache is not None
        hit = self.cache.get(key)
        if hit is not None:
            return LLMResponse(
                text=hit["text"],
                model=self.model,
                prompt_tokens=hit.get("prompt_tokens", 0),
                completion_tokens=hit.get("completion_tokens", 0),
                finish_reason=hit.get("finish_reason", "stop"),
                cached=True,
            )

        self.load()
        import torch  # noqa: PLC0415

        text = self._tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        enc = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=budget,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        # Only the continuation: decoding the whole sequence would hand the parser the
        # prompt back, and the prompt contains the word "Verdict".
        new = out[0][enc["input_ids"].shape[-1] :]
        answer = self._tokenizer.decode(new, skip_special_tokens=True).strip()

        payload = {
            "text": answer,
            "prompt_tokens": int(enc["input_ids"].shape[-1]),
            "completion_tokens": int(new.shape[-1]),
            "finish_reason": "length" if new.shape[-1] >= budget else "stop",
        }
        self.cache.set(key, payload)
        return LLMResponse(
            text=answer,
            model=self.model,
            prompt_tokens=payload["prompt_tokens"],
            completion_tokens=payload["completion_tokens"],
            finish_reason=payload["finish_reason"],
        )


def _adapter_fingerprint(adapter: Path | None) -> str:
    """Identify the weights, so retraining into the same path does not serve stale cache.

    Keyed on the adapter file's size and mtime rather than its contents: hashing a
    hundred megabytes on every call would cost more than the generation it saves.
    """
    if adapter is None:
        return "none"  # the base model's weights are pinned by its hub id
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        f = adapter / name
        if f.exists():
            st = f.stat()
            return f"{st.st_size}:{int(st.st_mtime)}"
    return "unknown"


def _base_from_adapter(adapter: Path | None) -> str | None:
    """Read the base model out of the adapter's own config, so it cannot be mismatched."""
    import json  # noqa: PLC0415

    if adapter is None:
        return None
    cfg = adapter / "adapter_config.json"
    if not cfg.exists():
        return None
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("base_model_name_or_path")
    except (json.JSONDecodeError, OSError):
        return None


def load_verifier(
    adapter: Path | str | None,
    cfg: RunConfig | None = None,
    base_only: bool = False,
) -> Any:
    """A verifier for `run_crag`, or None to mean "use the teacher".

    Three outcomes, because the G2 table needs all three:

    * `base_only` -- the student with no adapter, which is the "prompted 3B" row.
    * an adapter path -- the student with that adapter.
    * neither -- None, and the teacher answers, which is every Phase 1 result.

    Resolution order for the adapter is explicit path, then
    `VERIMEM_VERIFIER_ADAPTER`. A config asking for `verifier="finetuned"` with no
    adapter anywhere is an error rather than a silent fall back: that would report a
    fine-tuned row that was never fine-tuned.
    """
    path = adapter or os.environ.get("VERIMEM_VERIFIER_ADAPTER")
    if base_only:
        if path:
            raise SystemExit(
                "--verifier-base and --verifier-adapter are mutually exclusive: the "
                "first serves the student with no adapter, the second with one."
            )
        return FinetunedVerifier(None)
    if not path:
        if cfg is not None and cfg.verifier == "finetuned":
            raise SystemExit(
                'config asks for verifier="finetuned" but no adapter was given. '
                "Pass --verifier-adapter <dir> or set VERIMEM_VERIFIER_ADAPTER."
            )
        return None
    return FinetunedVerifier(Path(path))
